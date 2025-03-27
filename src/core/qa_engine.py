from typing import List, Dict, Any
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain

from src.schemas.main_schemas import DocumentChunk, QuestionResponse, DocumentReference

class QAEngine:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7
        )
        self.qa_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            verbose=True
        )

    async def answer_question(
        self,
        question: str,
        relevant_chunks: List[DocumentChunk],
        conversation_id: str = None
    ) -> QuestionResponse:
        """Generate an answer to a question using relevant document chunks."""
        # Convert chunks to LangChain documents
        documents = [
            Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    **chunk.metadata
                }
            )
            for chunk in relevant_chunks
        ]

        # Generate answer using the QA chain
        start_time = datetime.now()
        result = await self.qa_chain.arun(
            input_documents=documents,
            question=question
        )
        processing_time = (datetime.now() - start_time).total_seconds()

        # Create document references
        source_documents = [
            DocumentReference(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                similarity_score=1.0  # This should be updated with actual similarity scores
            )
            for chunk in relevant_chunks
        ]

        return QuestionResponse(
            answer=result,
            source_documents=source_documents,
            processing_time=processing_time,
            conversation_id=conversation_id
        )

    async def generate_follow_up_questions(
        self,
        question: str,
        answer: str,
        relevant_chunks: List[DocumentChunk]
    ) -> List[str]:
        """Generate follow-up questions based on the answer and context."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that generates relevant follow-up questions.
            Based on the original question, answer, and context, generate 3 follow-up questions
            that would help explore the topic further. Make the questions specific and focused."""),
            ("user", """Original Question: {question}
            Answer: {answer}
            Context: {context}
            
            Generate 3 follow-up questions.""")
        ])

        # Prepare context from chunks
        context = "\n".join([chunk.content for chunk in relevant_chunks])

        # Create and run the chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = await chain.arun(
            question=question,
            answer=answer,
            context=context
        )

        # Parse the result into a list of questions
        questions = [q.strip() for q in result.split("\n") if q.strip()]
        return questions[:3]  # Ensure we only return 3 questions 