from collections import defaultdict
from datetime import datetime
from typing import List

from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI

from src.core.api.dtos import DocumentChunk, DocumentReference, QuestionResponse
from src.core.impl.logging_config import setup_logger

logger = setup_logger("qa_engine")


class QAEngine:
    def __init__(self):
        logger.info("Initializing QAEngine")
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
        self.qa_chain = load_qa_chain(llm=self.llm, chain_type="stuff", verbose=True)
        # Store conversation history
        self.conversation_history = defaultdict(list)
        # Store document history per conversation
        self.conversation_documents = defaultdict(list)
        logger.info("QAEngine initialized successfully")

    async def answer_question(
        self,
        question: str,
        relevant_chunks: List[DocumentChunk],
        conversation_id: str = None,
    ) -> QuestionResponse:
        """Generate an answer to a question using relevant document chunks and conversation history."""
        logger.info(f"Processing question for conversation_id: {conversation_id}")
        logger.debug(f"Question: {question}")
        logger.debug(f"Number of relevant chunks: {len(relevant_chunks)}")

        # Convert chunks to LangChain documents
        documents = [
            Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    **chunk.metadata,
                },
            )
            for chunk in relevant_chunks
        ]

        # If conversation_id is provided, add previous context
        if conversation_id:
            logger.debug(
                f"Adding previous context for conversation_id: {conversation_id}"
            )
            # Add previous documents to current context
            previous_documents = self.conversation_documents[conversation_id]
            documents.extend(previous_documents)

            # Add previous Q&A pairs to the question for context
            if self.conversation_history[conversation_id]:
                history_context = "\nPrevious conversation:\n"
                for prev_q, prev_a in self.conversation_history[conversation_id]:
                    history_context += f"Q: {prev_q}\n{prev_a}\n"  # Removed A: prefix
                question = f"{history_context}\nCurrent question: {question}"

        # Generate answer using the QA chain
        start_time = datetime.now()
        logger.debug("Starting QA chain processing")
        result = await self.qa_chain.arun(input_documents=documents, question=question)
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"QA chain processing completed in {processing_time:.2f} seconds")

        # Create document references
        source_documents = [
            DocumentReference(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                similarity_score=1.0,  # This should be updated with actual similarity scores
            )
            for chunk in relevant_chunks
        ]

        # Update conversation history if conversation_id is provided
        if conversation_id:
            logger.debug(
                f"Updating conversation history for conversation_id: {conversation_id}"
            )
            self.conversation_history[conversation_id].append((question, result))
            # Update document history with new documents
            self.conversation_documents[conversation_id].extend(documents)

        logger.info(
            f"Successfully generated answer for conversation_id: {conversation_id}"
        )
        return QuestionResponse(
            answer=result,
            source_documents=source_documents,
            processing_time=processing_time,
            conversation_id=conversation_id,
        )

    async def generate_follow_up_questions(
        self, question: str, answer: str, relevant_chunks: List[DocumentChunk]
    ) -> List[str]:
        """Generate follow-up questions based on the answer and context."""
        logger.info("Generating follow-up questions")
        logger.debug(f"Original question: {question}")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant that generates relevant follow-up questions.
            Based on the original question, answer, and context, generate 3 follow-up questions
            that would help explore the topic further. Make the questions specific and focused.""",
                ),
                (
                    "user",
                    """Original Question: {question}
            Answer: {answer}
            Context: {context}
            
            Generate 3 follow-up questions.""",
                ),
            ]
        )

        # Prepare context from chunks
        context = "\n".join([chunk.content for chunk in relevant_chunks])

        # Create and run the chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        logger.debug("Starting follow-up questions generation")
        result = await chain.arun(question=question, answer=answer, context=context)

        # Parse the result into a list of questions
        questions = [q.strip() for q in result.split("\n") if q.strip()]
        questions = questions[:3]
        logger.info(f"Generated {len(questions)} follow-up questions")
        return questions
