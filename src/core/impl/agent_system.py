from datetime import datetime
from typing import List

from crewai import Agent, Crew, Process, Task

from src.core.api.dtos import (
    AgentAction,
    ComplexQueryRequest,
    ComplexQueryResponse,
    DocumentChunk,
    DocumentReference,
)
from src.core.impl.llm.base import LLMConfig
from src.core.impl.llm.factory import LLMFactory, LLMProvider
from src.core.impl.logging_config import setup_logger

logger = setup_logger("agent_system")


class AgentSystem:
    def __init__(self, llm_provider: LLMProvider = LLMProvider.OPENAI):
        logger.info(f"Initializing AgentSystem with LLM provider: {llm_provider}")
        self.llm_config = LLMConfig(
            model_name="gpt-4-turbo-preview",
            temperature=0.7,
            max_tokens=1000,
        )
        self.llm = LLMFactory.create(llm_provider, self.llm_config)
        logger.info("AgentSystem initialized successfully")

    def _create_researcher_agent(self) -> Agent:
        """Create a researcher agent specialized in finding relevant information."""
        logger.debug("Creating researcher agent")
        agent = Agent(
            role="Research Analyst",
            goal="Find and analyze relevant information from documents",
            backstory="""You are an expert research analyst with a keen eye for detail.
            Your job is to thoroughly analyze documents and extract relevant information
            to answer complex queries.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm.langchain_llm,
        )
        logger.debug("Researcher agent created successfully")
        return agent

    def _create_writer_agent(self) -> Agent:
        """Create a writer agent specialized in synthesizing information."""
        logger.debug("Creating writer agent")
        agent = Agent(
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
        logger.debug("Writer agent created successfully")
        return agent

    def _create_analyst_agent(self) -> Agent:
        """Create an analyst agent specialized in critical analysis."""
        logger.debug("Creating analyst agent")
        agent = Agent(
            role="Critical Analyst",
            goal="Analyze information critically and identify key insights",
            backstory="""You are a critical analyst with expertise in identifying
            patterns, connections, and implications in complex information.
            Your job is to provide deep insights and identify potential implications.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm.langchain_llm,
        )
        logger.debug("Analyst agent created successfully")
        return agent

    def _convert_chunks_to_references(
        self, chunks: List[DocumentChunk]
    ) -> List[DocumentReference]:
        """Convert DocumentChunk objects to DocumentReference objects."""
        logger.debug(f"Converting {len(chunks)} chunks to references")
        references = [
            DocumentReference(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                similarity_score=1.0,
            )
            for chunk in chunks
        ]
        logger.debug("Successfully converted chunks to references")
        return references

    async def process_complex_query(
        self, request: ComplexQueryRequest, relevant_chunks: List[DocumentChunk]
    ) -> ComplexQueryResponse:
        """Process a complex query using multiple specialized agents."""
        logger.info("Processing complex query")
        logger.debug(f"Query: {request.query}")
        logger.debug(f"Agent types: {request.agent_types}")
        logger.debug(f"Conversation ID: {request.conversation_id}")
        logger.debug(f"Number of relevant chunks: {len(relevant_chunks)}")

        start_time = datetime.now()
        agent_actions = []

        # Create agents based on request
        logger.debug("Creating agents based on request")
        agents = []
        if "researcher" in request.agent_types:
            agents.append(self._create_researcher_agent())
        if "writer" in request.agent_types:
            agents.append(self._create_writer_agent())
        if "analyst" in request.agent_types:
            agents.append(self._create_analyst_agent())

        if not agents:
            logger.info("No specific agent types requested, using default agents")
            agents = [self._create_researcher_agent(), self._create_writer_agent()]

        logger.info(f"Created {len(agents)} agents")

        # Prepare context from chunks
        logger.debug("Preparing context from chunks")
        context = "\n".join([chunk.content for chunk in relevant_chunks[:3]])

        # Create tasks for each agent
        logger.debug("Creating tasks for agents")
        tasks = []
        for i, agent in enumerate(agents):
            if i == 0:  # First agent (researcher)
                logger.debug("Creating research task")
                task = Task(
                    description=f"""Research the following query using the provided context:
                    Query: {request.query}
                    Context: {context}
                    
                    Focus on finding relevant information and key facts.""",
                    agent=agent,
                    expected_output="A comprehensive research summary with key findings and relevant information from the documents.",
                )
            elif i == 1:  # Second agent (writer)
                logger.debug("Creating writing task")
                task = Task(
                    description=f"""Based on the research findings, create a comprehensive response to:
                    Query: {request.query}
                    
                    Synthesize the information into a clear, well-structured answer.""",
                    agent=agent,
                    expected_output="A well-structured, comprehensive response that answers the query using the research findings.",
                )
            else:  # Additional agents (analyst)
                logger.debug("Creating analysis task")
                task = Task(
                    description=f"""Analyze the research findings and response for:
                    Query: {request.query}
                    
                    Provide critical insights and identify key implications.""",
                    agent=agent,
                    expected_output="Critical analysis and insights about the research findings and their implications.",
                )
            tasks.append(task)

        logger.info(f"Created {len(tasks)} tasks")

        # Create and run the crew
        logger.debug("Creating and running crew")
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )

        # Execute the crew's tasks
        logger.info("Starting crew execution")
        result = crew.kickoff()
        logger.info("Crew execution completed")

        # Record agent actions
        logger.debug("Recording agent actions")
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
        logger.info(f"Processing completed in {processing_time:.2f} seconds")

        # Convert chunks to references
        logger.debug("Converting chunks to references")
        document_references = self._convert_chunks_to_references(relevant_chunks)

        logger.info("Successfully processed complex query")
        return ComplexQueryResponse(
            answer=result,
            source_documents=document_references,
            processing_time=processing_time,
            conversation_id=request.conversation_id,
            agent_actions=agent_actions,
        )
