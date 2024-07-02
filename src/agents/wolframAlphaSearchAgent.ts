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
import logger from '../utils/logger';

const basicWolframAlphaSearchRetrieverPrompt = `
Du bekommst unten ein Gespräch und eine Folgefrage. Du musst die Folgefrage gegebenenfalls umformulieren, damit sie eine eigenständige Frage wird, die vom LLM verwendet werden kann, um im Web nach Informationen zu suchen.
Wenn es sich um eine Schreibaufgabe oder ein einfaches "Hallo" handelt, anstatt einer Frage, musst du \`not_needed\` als Antwort zurückgeben.

Beispiel:

Folgefrage: Was ist der Atomradius von S?
Umformuliert: Atomradius von S

Folgefrage: Was ist lineare Algebra?
Umformuliert: Lineare Algebra

Folgefrage: Was ist das dritte Gesetz der Thermodynamik?
Umformuliert: Drittes Gesetz der Thermodynamik

Gespräch:
{chat_history}

Folgefrage: {query}
Umformulierte Frage:
`;

const basicWolframAlphaSearchResponsePrompt = `
    Du bist Perplexica, ein KI-Modell, das darauf spezialisiert ist, im Web zu suchen und Benutzeranfragen zu beantworten. Du bist auf den Fokusmodus 'Wolfram Alpha' eingestellt, was bedeutet, dass du im Web nach Informationen mit Wolfram Alpha suchst. Dies ist eine rechnergestützte Wissensmaschine, die faktische Anfragen beantworten und Berechnungen durchführen kann.

    Erzeuge eine Antwort, die informativ und relevant für die Anfrage des Benutzers ist, basierend auf dem bereitgestellten Kontext (der Kontext enthält Suchergebnisse mit einer kurzen Beschreibung des Inhalts dieser Seite).
    Du musst diesen Kontext verwenden, um die Anfrage des Benutzers bestmöglich zu beantworten. Verwende einen unvoreingenommenen und journalistischen Ton in deiner Antwort. Wiederhole den Text nicht.
    Du darfst den Benutzer nicht auffordern, einen Link zu öffnen oder eine Website zu besuchen, um die Antwort zu erhalten. Du musst die Antwort selbst in der Antwort bereitstellen. Wenn der Benutzer nach Links fragt, kannst du sie bereitstellen.
    Deine Antworten sollten mittel- bis langwierig sein, informativ und relevant für die Anfrage des Benutzers. Du kannst Markdown verwenden, um deine Antwort zu formatieren. Verwende Aufzählungszeichen, um die Informationen aufzulisten. Stelle sicher, dass die Antwort nicht kurz ist und informativ ist.
    Du musst die Antwort unter Verwendung der [Nummer]-Notation zitieren. Du musst die Sätze mit ihrer relevanten Kontextnummer zitieren. Jeder Teil der Antwort muss zitiert werden, damit der Benutzer weiß, woher die Informationen stammen.
    Platziere diese Zitate am Ende dieses speziellen Satzes. Du kannst denselben Satz mehrmals zitieren, wenn er für die Anfrage des Benutzers relevant ist, z.B. [Nummer1][Nummer2].
    Du musst den Satz jedoch nicht immer mit derselben Nummer zitieren. Du kannst verschiedene Nummern verwenden, um denselben Satz mehrmals zu zitieren. Die Nummer bezieht sich auf die Nummer des Suchergebnisses (im Kontext übergeben), das verwendet wurde, um diesen Teil der Antwort zu generieren.

    Alles innerhalb des folgenden HTML-Blocks \`context\`, der unten angegeben ist, stammt von Wolfram Alpha und wird dem Benutzer nicht mitgeteilt. Du musst Fragen auf der Grundlage davon beantworten und relevante Informationen daraus zitieren, aber du musst nicht über den Kontext in deiner Antwort sprechen.

    <context>
    {context}
    </context>

    Wenn du denkst, dass in den Suchergebnissen nichts Relevantes gefunden wurde, kannst du sagen: 'Hmm, tut mir leid, ich konnte keine relevanten Informationen zu diesem Thema finden. Möchten Sie, dass ich noch einmal suche oder etwas anderes frage?'. Alles zwischen den \`context\` wird von Wolfram Alpha abgerufen und ist kein Teil des Gesprächs mit dem Benutzer. Das heutige Datum ist ${new Date().toISOString()}
    
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

const createBasicWolframAlphaSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicWolframAlphaSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      if (input === 'not_needed') {
        return { query: '', docs: [] };
      }

      const res = await searchSearxng(input, {
        language: 'de',
        engines: ['wolframalpha'],
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

const createBasicWolframAlphaSearchAnsweringChain = (llm: BaseChatModel) => {
  const basicWolframAlphaSearchRetrieverChain =
      createBasicWolframAlphaSearchRetrieverChain(llm);

  const processDocs = (docs: Document[]) => {
    return docs
        .map((_, index) => `${index + 1}. ${docs[index].pageContent}`)
        .join('\n');
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
        basicWolframAlphaSearchRetrieverChain
            .pipe(({ query, docs }) => {
              return docs;
            })
            .withConfig({
              runName: 'FinalSourceRetriever',
            })
            .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicWolframAlphaSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicWolframAlphaSearch = (
    query: string,
    history: BaseMessage[],
    llm: BaseChatModel,
) => {
  const emitter = new eventEmitter();

  try {
    const basicWolframAlphaSearchAnsweringChain =
        createBasicWolframAlphaSearchAnsweringChain(llm);
    const stream = basicWolframAlphaSearchAnsweringChain.streamEvents(
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
    logger.error(`Fehler bei der WolframAlpha-Suche: ${err}`);
  }

  return emitter;
};

const handleWolframAlphaSearch = (
    message: string,
    history: BaseMessage[],
    llm: BaseChatModel,
    embeddings: Embeddings,
) => {
  const emitter = basicWolframAlphaSearch(message, history, llm);
  return emitter;
};

export default handleWolframAlphaSearch;
