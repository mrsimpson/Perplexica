import { BaseMessage } from '@langchain/core/messages';
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import {
  RunnableSequence,
  RunnableMap,
  RunnableLambda,
} from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Document } from '@langchain/core/documents';
import { searchSearxng } from '../lib/searxng';
import type { StreamEvent } from '@langchain/core/tracers/log_stream';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import formatChatHistoryAsString from '../utils/formatHistory';
import eventEmitter from 'events';
import computeSimilarity from '../utils/computeSimilarity';
import logger from '../utils/logger';

const basicYoutubeSearchRetrieverPrompt = `
Du wirst unten eine Unterhaltung und eine Nachfrage erhalten. Du musst die Nachfrage bei Bedarf neu formulieren, damit sie eine eigenständige Frage darstellt, die vom LLM verwendet werden kann, um das Web nach Informationen zu durchsuchen.
Wenn es sich um eine Schreibaufgabe oder eine einfache Begrüßung handelt, anstatt um eine Frage, musst du als Antwort \`not_needed\` zurückgeben.

Beispiel:
1. Nachfrage: How does an A.C work?
Neu formuliert: A.C working

2. Nachfrage: Linear algebra explanation video
Neu formuliert: What is linear algebra?

3. Nachfrage: What is theory of relativity?
Neu formuliert: What is theory of relativity?

Unterhaltung:
{chat_history}

Nachfrage: {query}
Neu formulierte Frage:
`;

const basicYoutubeSearchResponsePrompt = `
    Du bist Perplexica, ein KI-Modell, das darauf spezialisiert ist, im Web zu suchen und Benutzeranfragen zu beantworten. Du bist derzeit im Fokusmodus 'Youtube' eingestellt, das bedeutet, du wirst nach Videos im Web suchen, die Youtube verwenden, und Informationen basierend auf dem Transkript des Videos bereitstellen.

    Generiere eine Antwort, die informativ und relevant für die Benutzeranfrage ist, basierend auf dem bereitgestellten Kontext (die Suchergebnisse enthalten eine kurze Beschreibung des Inhalts dieser Seite).
    Du musst diesen Kontext verwenden, um die Benutzeranfrage bestmöglich zu beantworten. Verwende einen unvoreingenommenen und journalistischen Ton in deiner Antwort. Wiederhole den Text nicht.
    Du darfst dem Benutzer nicht sagen, einen Link zu öffnen oder eine Website zu besuchen, um die Antwort zu erhalten. Du musst die Antwort in der Antwort selbst bereitstellen. Wenn der Benutzer nach Links fragt, kannst du sie bereitstellen.
    Deine Antworten sollten mittel bis lang sein, informativ und relevant für die Benutzeranfrage. Du kannst Markdown zur Formatierung deiner Antwort verwenden. Du solltest Aufzählungszeichen verwenden, um die Informationen aufzulisten. Stelle sicher, dass die Antwort nicht kurz ist und informativ ist.
    Du musst die Antwort mit der Notation [Nummer] zitieren. Du musst die Sätze mit ihrer relevanten Kontextnummer zitieren. Jeder Teil der Antwort muss zitiert werden, damit der Benutzer weiß, woher die Informationen stammen.
    Platziere diese Zitate am Ende des jeweiligen Satzes. Du kannst denselben Satz mehrmals zitieren, wenn dies für die Benutzeranfrage relevant ist, wie [Nummer1][Nummer2].
    Du musst ihn jedoch nicht mit derselben Nummer zitieren. Du kannst unterschiedliche Nummern verwenden, um denselben Satz mehrmals zu zitieren. Die Zahl bezieht sich auf die Nummer des Suchergebnisses (im Kontext übergeben), das verwendet wurde, um diesen Teil der Antwort zu generieren.

    Alles innerhalb des folgenden HTML-Blocks \`context\`, das von Youtube zurückgegeben wird, ist für dein Wissen bestimmt und wird vom Benutzer nicht geteilt. Du musst die Frage auf der Grundlage davon beantworten und die relevanten Informationen zitieren, aber du musst nicht über den Kontext in deiner Antwort sprechen. 

    <context>
    {context}
    </context>

    Wenn du denkst, dass in den Suchergebnissen nichts Relevantes vorhanden ist, kannst du sagen: 'Hmm, tut mir leid, ich konnte keine relevanten Informationen zu diesem Thema finden. Möchten Sie, dass ich erneut suche oder etwas anderes frage?'
    Alles zwischen dem \`context\` wird von Youtube abgerufen und ist kein Teil des Gesprächs mit dem Benutzer. Das heutige Datum ist ${new Date().toISOString()}
`;

const strParser = new StringOutputParser();

const handleStream = async (
    stream: AsyncGenerator<StreamEvent, any, unknown>,
    emitter: eventEmitter,
) => {
  for await (const event of stream) {
    if (
        event.event === 'on_chain_end' &&
        event.name === 'FinalSourceRetriever'
    ) {
      emitter.emit(
          'data',
          JSON.stringify({ type: 'sources', data: event.data.output }),
      );
    }
    if (
        event.event === 'on_chain_stream' &&
        event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit(
          'data',
          JSON.stringify({ type: 'response', data: event.data.chunk }),
      );
    }
    if (
        event.event === 'on_chain_end' &&
        event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit('end');
    }
  }
};

type BasicChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const createBasicYoutubeSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicYoutubeSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      if (input === 'not_needed') {
        return { query: '', docs: [] };
      }

      const res = await searchSearxng(input, {
        language: 'de',
        engines: ['youtube'],
      });

      const documents = res.results.map(
          (result) =>
              new Document({
                pageContent: result.content ? result.content : result.title,
                metadata: {
                  title: result.title,
                  url: result.url,
                  ...(result.img_src && { img_src: result.img_src }),
                },
              }),
      );

      return { query: input, docs: documents };
    }),
  ]);
};

const createBasicYoutubeSearchAnsweringChain = (
    llm: BaseChatModel,
    embeddings: Embeddings,
) => {
  const basicYoutubeSearchRetrieverChain =
      createBasicYoutubeSearchRetrieverChain(llm);

  const processDocs = async (docs: Document[]) => {
    return docs
        .map((_, index) => `${index + 1}. ${docs[index].pageContent}`)
        .join('\n');
  };

  const rerankDocs = async ({
                              query,
                              docs,
                            }: {
    query: string;
    docs: Document[];
  }) => {
    if (docs.length === 0) {
      return docs;
    }

    const docsWithContent = docs.filter(
        (doc) => doc.pageContent && doc.pageContent.length > 0,
    );

    const [docEmbeddings, queryEmbedding] = await Promise.all([
      embeddings.embedDocuments(docsWithContent.map((doc) => doc.pageContent)),
      embeddings.embedQuery(query),
    ]);

    const similarity = docEmbeddings.map((docEmbedding, i) => {
      const sim = computeSimilarity(queryEmbedding, docEmbedding);

      return {
        index: i,
        similarity: sim,
      };
    });

    const sortedDocs = similarity
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 15)
        .filter((sim) => sim.similarity > 0.3)
        .map((sim) => docsWithContent[sim.index]);

    return sortedDocs;
  };

  return RunnableSequence.from([
    RunnableMap.from({
      query: (input: BasicChainInput) => input.query,
      chat_history: (input: BasicChainInput) => input.chat_history,
      context: RunnableSequence.from([
        (input) => ({
          query: input.query,
          chat_history: formatChatHistoryAsString(input.chat_history),
        }),
        basicYoutubeSearchRetrieverChain
            .pipe(rerankDocs)
            .withConfig({
              runName: 'FinalSourceRetriever',
            })
            .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicYoutubeSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicYoutubeSearch = (
    query: string,
    history: BaseMessage[],
    llm: BaseChatModel,
    embeddings: Embeddings,
) => {
  const emitter = new eventEmitter();

  try {
    const basicYoutubeSearchAnsweringChain =
        createBasicYoutubeSearchAnsweringChain(llm, embeddings);

    const stream = basicYoutubeSearchAnsweringChain.streamEvents(
        {
          chat_history: history,
          query: query,
        },
        {
          version: 'v1',
        },
    );

    handleStream(stream, emitter);
  } catch (err) {
    emitter.emit(
        'error',
        JSON.stringify({ data: 'Ein Fehler ist aufgetreten. Bitte versuche es später erneut.' }),
    );
    logger.error(`Fehler bei der Youtube-Suche: ${err}`);
  }

  return emitter;
};

const handleYoutubeSearch = (
    message: string,
    history: BaseMessage[],
    llm: BaseChatModel,
    embeddings: Embeddings,
) => {
  const emitter = basicYoutubeSearch(message, history, llm, embeddings);
  return emitter;
};

export default handleYoutubeSearch;
