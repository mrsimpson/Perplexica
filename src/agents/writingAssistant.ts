import { BaseMessage } from '@langchain/core/messages';
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import type { StreamEvent } from '@langchain/core/tracers/log_stream';
import eventEmitter from 'events';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import logger from '../utils/logger';

const writingAssistantPrompt = `
Du bist Perplexica, ein KI-Modell, das darauf spezialisiert ist, im Web zu suchen und Benutzeranfragen zu beantworten. Du bist derzeit im Fokusmodus 'Writing Assistant' eingestellt, das bedeutet, du hilfst dem Benutzer dabei, auf eine gegebene Anfrage zu antworten. 
Da du ein Schreibassistent bist, führst du keine Websuchen durch. Wenn du der Meinung bist, dass dir Informationen fehlen, um die Anfrage zu beantworten, kannst du den Benutzer um weitere Informationen bitten oder vorschlagen, in einen anderen Fokusmodus zu wechseln.
Schreib bitte auf Deutsch.
`;

const strParser = new StringOutputParser();

const handleStream = async (
    stream: AsyncGenerator<StreamEvent, any, unknown>,
    emitter: eventEmitter,
) => {
  for await (const event of stream) {
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

const createWritingAssistantChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    ChatPromptTemplate.fromMessages([
      ['system', writingAssistantPrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const handleWritingAssistant = (
    query: string,
    history: BaseMessage[],
    llm: BaseChatModel,
    embeddings: Embeddings,
) => {
  const emitter = new eventEmitter();

  try {
    const writingAssistantChain = createWritingAssistantChain(llm);
    const stream = writingAssistantChain.streamEvents(
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
    logger.error(`Fehler im Schreibassistenten: ${err}`);
  }

  return emitter;
};

export default handleWritingAssistant;