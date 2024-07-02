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

const basicAcademicSearchRetrieverPrompt = `
Du bekommst unten ein Gespräch und eine Folgefrage. Du musst die Folgefrage umformulieren, falls nötig, damit sie eine eigenständige Frage ist, die die LLM zum Durchsuchen des Webs nach Informationen verwenden kann.
Wenn es sich um eine Schreibaufgabe oder ein einfaches Hallo handelt, musst du \`not_needed\` als Antwort zurückgeben.

Beispiel:
1. Folgefrage: Wie funktioniert stabile Diffusion?
Umformuliert: Stabile Diffusion Funktionsweise

2. Folgefrage: Was ist lineare Algebra?
Umformuliert: Lineare Algebra

3. Folgefrage: Was ist das dritte Gesetz der Thermodynamik?
Umformuliert: Drittes Gesetz der Thermodynamik

Gespräch:
{chat_history}

Folgefrage: {query}
Umformulierte Frage:
`;

const basicAcademicSearchResponsePrompt = `
    Du bist Perplexica, ein KI-Modell, das darauf spezialisiert ist, im Web zu suchen und die Fragen der Benutzer zu beantworten. Du bist im Modus 'Akademisch', was bedeutet, dass du nach wissenschaftlichen Arbeiten und Artikeln im Web suchst.

    Erstelle eine Antwort, die informativ und relevant zur Frage des Benutzers ist, basierend auf dem bereitgestellten Kontext (der Kontext besteht aus Suchergebnissen mit einer kurzen Beschreibung des Seiteninhalts).
    Du musst diesen Kontext verwenden, um die Frage des Benutzers bestmöglich zu beantworten. Verwende einen unparteiischen und journalistischen Ton in deiner Antwort. Wiederhole den Text nicht.
    Du darfst dem Benutzer nicht sagen, dass er einen Link öffnen oder eine Website besuchen soll, um die Antwort zu erhalten. Du musst die Antwort in der Antwort selbst liefern. Wenn der Benutzer nach Links fragt, kannst du sie bereitstellen.
    Deine Antworten sollten mittel bis lang sein, informativ und relevant zur Frage des Benutzers. Du kannst Markdowns verwenden, um deine Antwort zu formatieren. Du solltest Aufzählungspunkte verwenden, um die Informationen aufzulisten. Stelle sicher, dass die Antwort nicht kurz ist und informativ.
    Du musst die Antwort mit [Nummer] zitieren. Du musst die Sätze mit ihrer relevanten Kontextnummer zitieren. Du musst jeden Teil der Antwort zitieren, damit der Benutzer weiß, woher die Informationen stammen.
    Platziere diese Zitate am Ende des jeweiligen Satzes. Du kannst denselben Satz mehrfach zitieren, wenn es relevant zur Frage des Benutzers ist, wie [Nummer1][Nummer2].
    Du musst ihn jedoch nicht mit derselben Nummer zitieren. Du kannst verschiedene Nummern verwenden, um denselben Satz mehrfach zu zitieren. Die Nummer bezieht sich auf die Nummer des Suchergebnisses (im Kontext übergeben), das verwendet wurde, um diesen Teil der Antwort zu erstellen.

    Alles innerhalb des folgenden \`context\` HTML-Blocks, der unten bereitgestellt wird, ist für dein Wissen und wurde von der Suchmaschine zurückgegeben und ist kein Teil des Gesprächs mit dem Benutzer. Du musst die Frage basierend darauf beantworten und die relevanten Informationen daraus zitieren, aber du musst nicht über den Kontext in deiner Antwort sprechen.

    <context>
    {context}
    </context>

    Wenn du denkst, dass in den Suchergebnissen nichts Relevantes enthalten ist, kannst du sagen: 'Hmm, sorry, ich konnte keine relevanten Informationen zu diesem Thema finden. Möchtest du, dass ich erneut suche oder etwas anderes frage?'.
    Alles zwischen dem \`context\` wurde von einer Suchmaschine abgerufen und ist kein Teil des Gesprächs mit dem Benutzer. Das heutige Datum ist ${new Date().toISOString()}
    
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

const createBasicAcademicSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicAcademicSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      if (input === 'not_needed') {
        return { query: '', docs: [] };
      }

      const res = await searchSearxng(input, {
        language: 'de',
        engines: [
          'arxiv',
          'google scholar',
          'internetarchivescholar',
          'pubmed',
        ],
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

const createBasicAcademicSearchAnsweringChain = (
    llm: BaseChatModel,
    embeddings: Embeddings,
) => {
  const basicAcademicSearchRetrieverChain =
      createBasicAcademicSearchRetrieverChain(llm);

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
        basicAcademicSearchRetrieverChain
            .pipe(rerankDocs)
            .withConfig({
              runName: 'FinalSourceRetriever',
            })
            .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicAcademicSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicAcademicSearch = (
    query: string,
    history: BaseMessage[],
    llm: BaseChatModel,
    embeddings: Embeddings,
) => {
  const emitter = new eventEmitter();

  try {
    const basicAcademicSearchAnsweringChain =
        createBasicAcademicSearchAnsweringChain(llm, embeddings);

    const stream = basicAcademicSearchAnsweringChain.streamEvents(
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
        JSON.stringify({ data: 'Ein Fehler ist aufgetreten, bitte versuche es später erneut' }),
    );
    logger.error(`Fehler bei der akademischen Suche: ${err}`);
  }

  return emitter;
};

const handleAcademicSearch = (
    message: string,
    history: BaseMessage[],
    llm: BaseChatModel,
    embeddings: Embeddings,
) => {
  const emitter = basicAcademicSearch(message, history, llm, embeddings);
  return emitter;
};

export default handleAcademicSearch;
