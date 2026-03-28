import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import json
import logging
from pydantic import BaseModel, Field
from dataclasses import dataclass
from enum import Enum

# For Google ADK (Alternative: LangChain, CrewAI)
try:
    from google.adk.agents import LlmAgent, SequentialAgent
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    from google.genai import types
except ImportError:
    # Fallback for OpenAI
    from langchain.agents import AgentType, initialize_agent
    from langchain.llms import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_NAME = "investment_banking_advisor"
USER_ID = "default_user"

# ==================== ENUMS ====================
class TransactionType(str, Enum):
    M_AND_A = "Mergers & Acquisitions"
    IPO = "Initial Public Offering"
    DEBT = "Debt Issuance"
    EQUITY = "Equity Issuance"
    ADVISORY = "Strategic Advisory"

class CompanySize(str, Enum):
    SMALL_CAP = "Small Cap (< $500M)"
    MID_CAP = "Mid Cap ($500M - $5B)"
    LARGE_CAP = "Large Cap ($5B - $20B)"
    MEGA_CAP = "Mega Cap (> $20B)"

# ==================== PYDANTIC MODELS ====================
class CompanyMetrics(BaseModel):
    company_name: str = Field(..., description="Name of the company")
    market_cap: float = Field(..., description="Current market capitalization in millions")
    revenue: float = Field(..., description="Annual revenue in millions")
    ebitda: float = Field(..., description="EBITDA in millions")
    debt: float = Field(..., description="Total debt in millions")
    equity_value: float = Field(..., description="Equity value in millions")
    industry: str = Field(..., description="Industry sector")
    growth_rate: float = Field(..., description="Expected growth rate (%)")

class Valuation(BaseModel):
    dcf_value: float = Field(..., description="DCF valuation in millions")
    comparable_multiples: Dict[str, float] = Field(..., description="Valuation multiples (EV/EBITDA, P/E, etc.)")
    precedent_transactions: float = Field(..., description="Precedent transaction value")
    fair_value_range: tuple = Field(..., description="Fair value range (low, high)")
    recommendation: str = Field(..., description="Valuation recommendation")

class TransactionOpportunity(BaseModel):
    target_company: str = Field(..., description="Target company name")
    transaction_type: str = Field(..., description="Type of transaction")
    deal_value: float = Field(..., description="Expected deal value in millions")
    synergies: float = Field(..., description="Estimated synergies in millions")
    strategic_rationale: str = Field(..., description="Strategic rationale")
    risks: List[str] = Field(..., description="Key risks")
    timeline: int = Field(..., description="Expected timeline in months")

class FinancialProjection(BaseModel):
    year: int = Field(..., description="Year number")
    revenue: float = Field(..., description="Projected revenue")
    ebitda: float = Field(..., description="Projected EBITDA")
    fcf: float = Field(..., description="Free cash flow")
    growth_rate: float = Field(..., description="YoY growth rate (%)")

class InvestmentThesis(BaseModel):
    company_name: str = Field(..., description="Company name")
    investment_rationale: str = Field(..., description="Investment rationale")
    key_drivers: List[str] = Field(..., description="Key value drivers")
    financial_projections: List[FinancialProjection] = Field(..., description="5-year projections")
    entry_valuation: float = Field(..., description="Recommended entry valuation")
    exit_strategy: str = Field(..., description="Exit strategy")

class CompetitiveAnalysis(BaseModel):
    target_company: str = Field(..., description="Target company")
    competitors: List[Dict[str, Any]] = Field(..., description="Competitor analysis")
    market_position: str = Field(..., description="Market positioning")
    competitive_advantages: List[str] = Field(..., description="Competitive advantages")
    market_share: float = Field(..., description="Estimated market share (%)")

class DealStructure(BaseModel):
    deal_type: str = Field(..., description="Deal structure type (All cash, Stock, Mixed)")
    purchase_price: float = Field(..., description="Total purchase price in millions")
    debt_financing: float = Field(..., description="Debt financing component")
    equity_contribution: float = Field(..., description="Equity contribution")
    earnouts: Optional[float] = Field(None, description="Earnout provisions")
    contingencies: List[str] = Field(..., description="Key contingencies")

class ValuationAnalysis(BaseModel):
    company_metrics: CompanyMetrics = Field(..., description="Company metrics")
    valuations: Valuation = Field(..., description="Valuation analysis")
    competitive_analysis: CompetitiveAnalysis = Field(..., description="Competitive landscape")
    investment_thesis: InvestmentThesis = Field(..., description="Investment thesis")

class TransactionAnalysis(BaseModel):
    opportunities: List[TransactionOpportunity] = Field(..., description="Transaction opportunities")
    deal_structure: DealStructure = Field(..., description="Recommended deal structure")
    sources_and_uses: Dict[str, float] = Field(..., description="Sources & Uses of funds")
    pro_forma_impact: Dict[str, float] = Field(..., description="Pro forma impact")
    strategic_rationale: str = Field(..., description="Strategic rationale")

class ExecutionPlan(BaseModel):
    milestones: List[Dict[str, Any]] = Field(..., description="Key milestones")
    timeline: List[str] = Field(..., description="Project timeline")
    dependencies: List[str] = Field(..., description="Critical dependencies")
    risk_mitigation: List[str] = Field(..., description="Risk mitigation strategies")
    key_success_factors: List[str] = Field(..., description="Key success factors")

# ==================== AGENT SYSTEM ====================
class InvestmentBankingAdvisor:
    def __init__(self):
        self.session_service = InMemorySessionService()
        
        # Agent 1: Valuation Analysis Agent
        self.valuation_agent = LlmAgent(
            name="ValuationAnalysisAgent",
            model="gemini-2.5-flash",
            description="Performs comprehensive valuation analysis using DCF, comparable multiples, and precedent transactions",
            instruction="""You are a Senior Valuation Analyst at a leading investment bank.
            
Your responsibilities:
1. Analyze company financials and metrics
2. Perform DCF valuation with detailed assumptions
3. Calculate EV/EBITDA, P/E, and other relevant multiples
4. Benchmark against comparable companies
5. Analyze precedent transactions
6. Provide valuation ranges and fair value assessments
7. Identify key value drivers and sensitivities
8. Conduct competitive analysis

Key considerations:
- Industry dynamics and market position
- Growth prospects and margin trends
- Capital structure and cash flow generation
- Macroeconomic factors and cyclicality
- Regulatory environment
- Management quality and execution track record

Provide detailed, institutional-quality analysis with specific numbers and percentages.
Store your analysis in state['valuation_analysis'].""",
            output_schema=ValuationAnalysis,
            output_key="valuation_analysis"
        )
        
        # Agent 2: Transaction Strategy Agent
        self.transaction_agent = LlmAgent(
            name="TransactionStrategyAgent",
            model="gemini-2.5-flash",
            description="Identifies transaction opportunities and designs optimal deal structures",
            instruction="""You are a Managing Director specializing in M&A strategy.

Your responsibilities:
1. Identify strategic transaction opportunities
2. Calculate deal value and synergy potential
3. Design optimal deal structures (cash vs stock vs mixed)
4. Prepare sources and uses analysis
5. Model pro forma financial impact
6. Assess integration requirements
7. Identify key risks and mitigation strategies
8. Define success metrics and value creation levers

Key considerations:
- Strategic fit and synergy potential
- Financing availability and cost
- Regulatory and antitrust implications
- Cultural and operational integration
- Tax optimization
- Stakeholder considerations
- Market conditions and timing

Read state['valuation_analysis'] for pricing reference.
Store your recommendations in state['transaction_analysis'].""",
            output_schema=TransactionAnalysis,
            output_key="transaction_analysis"
        )
        
        # Agent 3: Investment Thesis Agent
        self.thesis_agent = LlmAgent(
            name="InvestmentThesisAgent",
            model="gemini-2.5-flash",
            description="Develops comprehensive investment theses with detailed financial models",
            instruction="""You are a Portfolio Manager and Investment Strategist.

Your responsibilities:
1. Develop comprehensive investment thesis
2. Create detailed 5-year financial projections
3. Identify key value creation drivers
4. Model different exit scenarios
5. Calculate potential returns (IRR, MOIC)
6. Assess downside scenarios and sensitivities
7. Define critical assumptions
8. Recommend entry and exit strategies

Key considerations:
- Market trends and secular tailwinds/headwinds
- Management capability to execute
- Competitive positioning and barriers to entry
- Capital allocation discipline
- Dividend and return of capital potential
- Market comparables and market cap trajectory
- Inflation and cost pressures

Build on state['valuation_analysis'] and state['transaction_analysis'].
Store your thesis in state['investment_thesis'].""",
            output_schema=InvestmentThesis,
            output_key="investment_thesis"
        )
        
        # Agent 4: Execution Agent
        self.execution_agent = LlmAgent(
            name="ExecutionPlanAgent",
            model="gemini-2.5-flash",
            description="Develops detailed execution plans and risk management strategies",
            instruction="""You are a Managing Director overseeing transaction execution.

Your responsibilities:
1. Create detailed project timeline and milestones
2. Identify critical path items and dependencies
3. Develop risk assessment and mitigation plans
4. Define integration roadmap
5. Create regulatory and stakeholder management strategy
6. Establish governance and decision-making framework
7. Identify resources and talent requirements
8. Set success metrics and KPIs

Key considerations:
- Antitrust and regulatory requirements
- Financial reporting and disclosure obligations
- Debt covenant considerations
- Insurance and representations & warranties
- Retention and incentive alignment
- Communications and stakeholder management
- Operational synergy realization
- Technology and system integration

Reference all previous analyses.
Store your execution plan in state['execution_plan'].""",
            output_schema=ExecutionPlan,
            output_key="execution_plan"
        )
        
        # Coordinator Agent
        self.coordinator_agent = SequentialAgent(
            name="InvestmentBankingCoordinator",
            description="Orchestrates comprehensive investment banking analysis",
            sub_agents=[
                self.valuation_agent,
                self.transaction_agent,
                self.thesis_agent,
                self.execution_agent
            ]
        )
        
        self.runner = Runner(
            agent=self.coordinator_agent,
            app_name=APP_NAME,
            session_service=self.session_service
        )

    async def analyze_opportunity(self, opportunity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze an investment opportunity comprehensively"""
        session_id = f"ib_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            initial_state = {
                "company_metrics": opportunity_data.get("company_metrics", {}),
                "transaction_type": opportunity_data.get("transaction_type"),
                "financial_data": opportunity_data.get("financial_data", {}),
                "market_data": opportunity_data.get("market_data", {}),
                "strategic_context": opportunity_data.get("strategic_context", "")
            }
            
            session = self.session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
                state=initial_state
            )
            
            user_content = types.Content(
                role='user',
                parts=[types.Part(text=json.dumps(opportunity_data))]
            )
            
            async for event in self.runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=user_content
            ):
                if event.is_final_response() and event.author == self.coordinator_agent.name:
                    break
            
            updated_session = self.session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )
            
            results = {}
            for key in ["valuation_analysis", "transaction_analysis", "investment_thesis", "execution_plan"]:
                value = updated_session.state.get(key)
                results[key] = json.loads(value) if isinstance(value, str) else value
            
            return results
            
        except Exception as e:
            logger.exception(f"Error during analysis: {str(e)}")
            raise
        finally:
            self.session_service.delete_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )

# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(
        page_title="Investment Banking Advisor",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with st.sidebar:
        st.title("🏦 Setup & Configuration")
        st.info("Configure your investment banking analysis parameters")
        
    st.title("💼 Investment Banking Analysis Platform")
    st.caption("Institutional-grade M&A and Investment Analysis")
    
    input_tab, results_tab, about_tab = st.tabs([
        "📊 Analysis Input",
        "📈 Results",
        "ℹ️ About"
    ])
    
    with input_tab:
        st.header("Investment Opportunity Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Company Information")
            company_name = st.text_input("Company Name", "TechCorp Inc.")
            market_cap = st.number_input("Market Cap ($M)", 0.0, 100000.0, 5000.0)
            revenue = st.number_input("Annual Revenue ($M)", 0.0, 50000.0, 1000.0)
            ebitda = st.number_input("EBITDA ($M)", 0.0, 10000.0, 200.0)
            
        with col2:
            st.subheader("💰 Financial Metrics")
            debt = st.number_input("Total Debt ($M)", 0.0, 20000.0, 500.0)
            equity_value = st.number_input("Equity Value ($M)", 0.0, 20000.0, 4500.0)
            industry = st.selectbox("Industry", [
                "Technology", "Healthcare", "Financials",
                "Consumer", "Industrial", "Energy", "Other"
            ])
            growth_rate = st.number_input("Expected Growth Rate (%)", -20.0, 100.0, 15.0)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🤝 Transaction Details")
            transaction_type = st.selectbox("Transaction Type", [
                "Mergers & Acquisitions",
                "Initial Public Offering",
                "Debt Issuance",
                "Equity Issuance",
                "Strategic Advisory"
            ])
            
        with col4:
            st.subheader("📍 Strategic Context")
            strategic_context = st.text_area(
                "Strategic Rationale",
                "Enter the strategic context and investment thesis...",
                height=100
            )
        
        if st.button("🔄 Run Comprehensive Analysis", use_container_width=True):
            opportunity_data = {
                "company_metrics": {
                    "company_name": company_name,
                    "market_cap": market_cap,
                    "revenue": revenue,
                    "ebitda": ebitda,
                    "debt": debt,
                    "equity_value": equity_value,
                    "industry": industry,
                    "growth_rate": growth_rate
                },
                "transaction_type": transaction_type,
                "financial_data": {
                    "revenue": revenue,
                    "ebitda": ebitda,
                    "debt": debt
                },
                "market_data": {
                    "market_cap": market_cap,
                    "industry": industry
                },
                "strategic_context": strategic_context
            }
            
            with st.spinner("🤖 AI agents analyzing opportunity..."):
                advisor = InvestmentBankingAdvisor()
                try:
                    results = asyncio.run(advisor.analyze_opportunity(opportunity_data))
                    st.session_state.analysis_results = results
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with results_tab:
        if "analysis_results" in st.session_state:
            results = st.session_state.analysis_results
            
            result_tabs = st.tabs([
                "💎 Valuation",
                "🤝 Transaction",
                "📊 Investment Thesis",
                "⚙️ Execution"
            ])
            
            with result_tabs[0]:
                st.subheader("Valuation Analysis")
                if "valuation_analysis" in results:
                    st.json(results["valuation_analysis"])
            
            with result_tabs[1]:
                st.subheader("Transaction Strategy")
                if "transaction_analysis" in results:
                    st.json(results["transaction_analysis"])
            
            with result_tabs[2]:
                st.subheader("Investment Thesis")
                if "investment_thesis" in results:
                    st.json(results["investment_thesis"])
            
            with result_tabs[3]:
                st.subheader("Execution Plan")
                if "execution_plan" in results:
                    st.json(results["execution_plan"])
        else:
            st.info("Run analysis to see results here")
    
    with about_tab:
        st.markdown("""
        ### Investment Banking Analysis Platform
        
        This platform provides institutional-grade analysis across four key pillars:
        
        #### 1. **Valuation Analysis Agent**
        - DCF modeling with scenario analysis
        - Comparable company multiples
        - Precedent transaction analysis
        - Fair value range assessment
        
        #### 2. **Transaction Strategy Agent**
        - Deal structure optimization
        - Synergy quantification
        - Sources & Uses analysis
        - Pro forma integration modeling
        
        #### 3. **Investment Thesis Agent**
        - 5-year financial projections
        - IRR and MOIC analysis
        - Value creation drivers
        - Exit strategy modeling
        
        #### 4. **Execution Plan Agent**
        - Project timeline and milestones
        - Risk assessment and mitigation
        - Integration roadmap
        - Success metrics and KPIs
        
        ### Technology Stack
        
        - **LLM**: Google Gemini 2.5 Flash (or OpenAI GPT-4)
        - **Framework**: Google ADK / LangChain
        - **UI**: Streamlit
        - **Data**: Pandas, NumPy
        - **Visualization**: Plotly
        
        ### Key Features
        
        ✅ Multi-agent collaborative analysis
        ✅ Structured output schemas (Pydantic)
        ✅ Stateful agent communication
        ✅ Real-time processing
        ✅ Institutional-grade recommendations
        """)

if __name__ == "__main__":
    load_dotenv()
    main()
