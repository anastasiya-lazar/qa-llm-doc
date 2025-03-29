from datetime import datetime
from typing import List

from crewai import Agent, Crew, Process, Task

from src.core.api.dtos import (AgentAction, ComplexQueryRequest,
                               ComplexQueryResponse, DocumentChunk,
                               DocumentReference)
from src.core.impl.llm.factory import LLMFactory, LLMProvider
from src.core.impl.llm.base import LLMConfig


class AgentSystem:
    def __init__(self, llm_provider: LLMProvider = LLMProvider.OPENAI):
        self.llm_config = LLMConfig(
            model_name="gpt-4-turbo-preview",
            temperature=0.7,
            max_tokens=1000,
        )
        self.llm = LLMFactory.create(llm_provider, self.llm_config)

    def _create_researcher_agent(self) -> Agent:
        """Create a researcher agent specialized in finding relevant information."""
        return Agent(
            role="Research Analyst",
            goal="Find and analyze relevant information from documents",
            backstory="""You are an expert research analyst with a keen eye for detail.
            Your job is to thoroughly analyze documents and extract relevant information
            to answer complex queries.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm.langchain_llm,
        )

    def _create_writer_agent(self) -> Agent:
        """Create a writer agent specialized in synthesizing information."""
        return Agent(
            role="Content Writer",
            goal="Synthesize information into clear, coherent responses",
            backstory="""You are a skilled content writer with expertise in
            synthesizing complex information into clear, concise explanations.
            Your job is to take research findings and create well-structured,
            comprehensive responses.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm.langchain_llm,
        )

    def _create_analyst_agent(self) -> Agent:
        """Create an analyst agent specialized in critical analysis."""
        return Agent(
            role="Critical Analyst",
            goal="Analyze information critically and identify key insights",
            backstory="""You are a critical analyst with expertise in identifying
            patterns, connections, and implications in complex information.
            Your job is to provide deep insights and identify potential implications.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm.langchain_llm,
        )

    def _convert_chunks_to_references(
        self, chunks: List[DocumentChunk]
    ) -> List[DocumentReference]:
        """Convert DocumentChunk objects to DocumentReference objects."""
        return [
            DocumentReference(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                similarity_score=1.0,
            )
            for chunk in chunks
        ]

    async def process_complex_query(
        self, request: ComplexQueryRequest, relevant_chunks: List[DocumentChunk]
    ) -> ComplexQueryResponse:
        """Process a complex query using multiple specialized agents."""
        start_time = datetime.now()
        agent_actions = []

        # Create agents based on request
        agents = []
        if "researcher" in request.agent_types:
            agents.append(self._create_researcher_agent())
        if "writer" in request.agent_types:
            agents.append(self._create_writer_agent())
        if "analyst" in request.agent_types:
            agents.append(self._create_analyst_agent())

        if not agents:
            agents = [self._create_researcher_agent(), self._create_writer_agent()]

        # Prepare context from chunks
        context = "\n".join(
            [chunk.content for chunk in relevant_chunks[:3]]
        )

        # Create tasks for each agent
        tasks = []
        for i, agent in enumerate(agents):
            if i == 0:  # First agent (researcher)
                task = Task(
                    description=f"""Research the following query using the provided context:
                    Query: {request.query}
                    Context: {context}
                    
                    Focus on finding relevant information and key facts.""",
                    agent=agent,
                    expected_output="A comprehensive research summary with key findings and relevant information from the documents.",
                )
            elif i == 1:  # Second agent (writer)
                task = Task(
                    description=f"""Based on the research findings, create a comprehensive response to:
                    Query: {request.query}
                    
                    Synthesize the information into a clear, well-structured answer.""",
                    agent=agent,
                    expected_output="A well-structured, comprehensive response that answers the query using the research findings.",
                )
            else:  # Additional agents (analyst)
                task = Task(
                    description=f"""Analyze the research findings and response for:
                    Query: {request.query}
                    
                    Provide critical insights and identify key implications.""",
                    agent=agent,
                    expected_output="Critical analysis and insights about the research findings and their implications.",
                )
            tasks.append(task)

        # Create and run the crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )

        # Execute the crew's tasks
        result = crew.kickoff()

        # Record agent actions
        for agent, task in zip(agents, tasks):
            agent_actions.append(
                AgentAction(
                    agent_name=agent.role,
                    action="task_execution",
                    input=task.description,
                    output=result,
                    timestamp=datetime.now(),
                )
            )

        processing_time = (datetime.now() - start_time).total_seconds()

        # Convert chunks to references
        document_references = self._convert_chunks_to_references(relevant_chunks)

        return ComplexQueryResponse(
            answer=result,
            source_documents=document_references,
            processing_time=processing_time,
            conversation_id=request.conversation_id,
            agent_actions=agent_actions,
        )
