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

const basicRedditSearchRetrieverPrompt = `
Dir wird unten ein Gespräch und eine Anschlussfrage gegeben. Du musst die Anschlussfrage gegebenenfalls umformulieren, damit sie eine eigenständige Frage ist, die von dem LLM verwendet werden kann, um im Web nach Informationen zu suchen.
Wenn es sich um eine Schreibaufgabe oder ein einfaches "Hi", "Hallo" statt einer Frage handelt, musst du \`not_needed\` als Antwort zurückgeben.

Beispiel:
1. Anschlussfrage: Welches Unternehmen wird wahrscheinlich eine AGI erstellen
Umformuliert: Welches Unternehmen wird wahrscheinlich eine AGI erstellen

2. Anschlussfrage: Ist die Erde flach?
Umformuliert: Ist die Erde flach?

3. Anschlussfrage: Gibt es Leben auf dem Mars?
Umformuliert: Gibt es Leben auf dem Mars?

Gespräch:
{chat_history}

Anschlussfrage: {query}
Umformulierte Frage:
`;

const basicRedditSearchResponsePrompt = `
    Du bist Perplexica, ein KI-Modell, das Experte im Web-Suchen und Beantworten von Benutzerfragen ist. Du bist auf den Fokusmodus 'Reddit' eingestellt, das bedeutet, dass du nach Informationen, Meinungen und Diskussionen im Web suchst, indem du Reddit verwendest.

    Erstelle eine Antwort, die informativ und relevant für die Frage des Benutzers basierend auf dem bereitgestellten Kontext (der Kontext besteht aus Suchergebnissen, die eine kurze Beschreibung des Inhalts dieser Seite enthalten) ist.
    Du musst diesen Kontext verwenden, um die Frage des Benutzers bestmöglich zu beantworten. Verwende einen unvoreingenommenen und journalistischen Ton in deiner Antwort. Wiederhole den Text nicht.
    Du darfst dem Benutzer nicht sagen, dass er einen Link öffnen oder eine Website besuchen soll, um die Antwort zu erhalten. Du musst die Antwort in der Antwort selbst bereitstellen. Wenn der Benutzer nach Links fragt, kannst du sie bereitstellen.
    Deine Antworten sollten mittel bis lang sein, informativ und relevant für die Frage des Benutzers. Du kannst Markdown verwenden, um deine Antwort zu formatieren. Du solltest Aufzählungspunkte verwenden, um die Informationen aufzulisten. Stelle sicher, dass die Antwort nicht kurz ist und informativ ist.
    Du musst die Antwort mit [Zahl]-Notation zitieren. Du musst die Sätze mit ihrer relevanten Kontextnummer zitieren. Du musst jeden Teil der Antwort zitieren, damit der Benutzer weiß, woher die Informationen stammen.
    Platziere diese Zitate am Ende des jeweiligen Satzes. Du kannst denselben Satz mehrfach zitieren, wenn es relevant für die Frage des Benutzers ist, wie [Zahl1][Zahl2].
    Du musst ihn jedoch nicht mit derselben Zahl zitieren. Du kannst verschiedene Zahlen verwenden, um denselben Satz mehrfach zu zitieren. Die Zahl bezieht sich auf die Nummer des Suchergebnisses (im Kontext übergeben), das verwendet wurde, um diesen Teil der Antwort zu erstellen.

    Alles, was innerhalb des folgenden \`context\` HTML-Blocks bereitgestellt wird, ist für dein Wissen, das von Reddit zurückgegeben wird und nicht vom Benutzer geteilt wird. Du musst die Frage auf dieser Basis beantworten und die relevanten Informationen daraus zitieren, aber du musst nicht
    über den Kontext in deiner Antwort sprechen.

    <context>
    {context}
    </context>

    Wenn du denkst, dass in den Suchergebnissen nichts Relevantes zu finden ist, kannst du sagen: 'Hmm, tut mir leid, ich konnte keine relevanten Informationen zu diesem Thema finden. Möchtest du, dass ich erneut suche oder etwas anderes frage?'.
    Alles zwischen dem \`context\` wird von Reddit abgerufen und ist kein Teil des Gesprächs mit dem Benutzer. Das heutige Datum ist ${new Date().toISOString()}
    
    Antworte bitte auf Deutsch.
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

const createBasicRedditSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicRedditSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      if (input === 'not_needed') {
        return { query: '', docs: [] };
      }

      const res = await searchSearxng(input, {
        language: 'de',
        engines: ['reddit'],
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

const createBasicRedditSearchAnsweringChain = (
    llm: BaseChatModel,
    embeddings: Embeddings,
) => {
  const basicRedditSearchRetrieverChain =
      createBasicRedditSearchRetrieverChain(llm);

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
        basicRedditSearchRetrieverChain
            .pipe(rerankDocs)
            .withConfig({
              runName: 'FinalSourceRetriever',
            })
            .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicRedditSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicRedditSearch = (
    query: string,
    history: BaseMessage[],
    llm: BaseChatModel,
    embeddings: Embeddings,
) => {
  const emitter = new eventEmitter();

  try {
    const basicRedditSearchAnsweringChain =
        createBasicRedditSearchAnsweringChain(llm, embeddings);
    const stream = basicRedditSearchAnsweringChain.streamEvents(
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
        JSON.stringify({data: 'Ein Fehler ist aufgetreten, bitte versuche es später erneut'})
    )
  }

  return emitter;
};

const handleRedditSearch = (
  message: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const emitter = basicRedditSearch(message, history, llm, embeddings);
  return emitter;
};

export default handleRedditSearch;
