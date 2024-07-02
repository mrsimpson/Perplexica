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

const basicSearchRetrieverPrompt = `
Dir wird unten ein Gespräch und eine Folgefrage gegeben. Du musst die Folgefrage gegebenenfalls umformulieren, damit sie eine eigenständige Frage ist, die von der LLM verwendet werden kann, um im Web nach Informationen zu suchen.
    Wenn es sich um eine Schreibaufgabe oder ein einfaches Hallo handelt und nicht um eine Frage, musst du \`not_needed\` als Antwort zurückgeben.

Beispiel:
1. Folgefrage: Was ist die Hauptstadt von Frankreich?
Umformuliert: Hauptstadt von Frankreich

2. Folgefrage: Wie viele Einwohner hat New York City?
Umformuliert: Einwohnerzahl von New York City

3. Folgefrage: Was ist Docker?
Umformuliert: Was ist Docker

Gespräch:
{chat_history}

Folgefrage: {query}
Umformulierte Frage:
`;

const basicWebSearchResponsePrompt = `
    Du bist Perplexica, ein KI-Modell, das Experte im Web-Suchen und Beantworten von Benutzeranfragen ist.

    Erzeuge eine Antwort, die informativ und relevant für die Frage des Benutzers ist, basierend auf dem bereitgestellten Kontext (der Kontext besteht aus Suchergebnissen mit einer kurzen Beschreibung des Inhalts der Seite).
    Du musst diesen Kontext verwenden, um die Frage des Benutzers bestmöglich zu beantworten. Verwende einen unparteiischen und journalistischen Ton in deiner Antwort. Wiederhole den Text nicht.
    Du darfst dem Benutzer nicht sagen, dass er einen Link öffnen oder eine Website besuchen soll, um die Antwort zu erhalten. Du musst die Antwort selbst in der Antwort geben. Wenn der Benutzer nach Links fragt, kannst du sie bereitstellen.
    Deine Antworten sollten mittel bis lang sein, informativ und relevant für die Frage des Benutzers. Du kannst Markdown verwenden, um deine Antwort zu formatieren. Du solltest Aufzählungspunkte verwenden, um die Informationen aufzulisten. Stelle sicher, dass die Antwort nicht kurz ist und informativ ist.
    Du musst die Antwort mit der [Nummer] Notation zitieren. Du musst die Sätze mit ihrem relevanten Kontext zitieren, damit der Benutzer weiß, woher die Informationen stammen.
    Setze diese Zitate am Ende des jeweiligen Satzes. Du kannst denselben Satz mehrfach zitieren, wenn es relevant für die Frage des Benutzers ist, wie [Nummer1][Nummer2].
    Du musst es jedoch nicht mit derselben Nummer zitieren. Du kannst verschiedene Nummern verwenden, um denselben Satz mehrfach zu zitieren. Die Nummer bezieht sich auf die Nummer des Suchergebnisses (im Kontext angegeben), das verwendet wurde, um diesen Teil der Antwort zu erstellen.

    Alles innerhalb des folgenden \`context\` HTML-Blocks ist für dein Wissen, das von der Suchmaschine zurückgegeben wurde, und wird nicht mit dem Benutzer geteilt. Du musst die Frage auf dieser Basis beantworten und die relevanten Informationen daraus zitieren, aber du musst nicht über den Kontext in deiner Antwort sprechen. 

    <context>
    {context}
    </context>

    Wenn du denkst, dass es keine relevanten Informationen in den Suchergebnissen gibt, kannst du sagen: 'Hmm, sorry, ich konnte keine relevanten Informationen zu diesem Thema finden. Möchtest du, dass ich erneut suche oder etwas anderes frage?'.
    Alles zwischen dem \`context\` ist von einer Suchmaschine abgerufen und nicht Teil des Gesprächs mit dem Benutzer. Das heutige Datum ist ${new Date().toISOString()}
    
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

const createBasicWebSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      if (input === 'not_needed') {
        return { query: '', docs: [] };
      }

      const res = await searchSearxng(input, {
        language: 'de',
      });

      const documents = res.results.map(
          (result) =>
              new Document({
                pageContent: result.content,
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

const createBasicWebSearchAnsweringChain = (
    llm: BaseChatModel,
    embeddings: Embeddings,
) => {
  const basicWebSearchRetrieverChain = createBasicWebSearchRetrieverChain(llm);

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
        .filter((sim) => sim.similarity > 0.5)
        .slice(0, 15)
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
        basicWebSearchRetrieverChain
            .pipe(rerankDocs)
            .withConfig({
              runName: 'FinalSourceRetriever',
            })
            .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicWebSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicWebSearch = (
    query: string,
    history: BaseMessage[],
    llm: BaseChatModel,
    embeddings: Embeddings,
) => {
  const emitter = new eventEmitter();

  try {
    const basicWebSearchAnsweringChain = createBasicWebSearchAnsweringChain(
        llm,
        embeddings,
    );

    const stream = basicWebSearchAnsweringChain.streamEvents(
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
        JSON.stringify({ data: 'Ein Fehler ist aufgetreten, bitte versuche es später noch einmal' }),
    );
    logger.error(`Fehler bei der Websuche: ${err}`);
  }

  return emitter;
};

const handleWebSearch = (
    message: string,
    history: BaseMessage[],
    llm: BaseChatModel,
    embeddings: Embeddings,
) => {
  const emitter = basicWebSearch(message, history, llm, embeddings);
  return emitter;
};

export default handleWebSearch;