from typing import Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Evidence bundle (from extractor — not validated by Pydantic, just typed)
# ---------------------------------------------------------------------------

class EvidenceBundle(BaseModel):
    source_url: str
    page_title: str = ""
    meta_description: str = ""
    headings: list[str] = Field(default_factory=list)
    paragraphs: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)
    tables: list[list[list[str]]] = Field(default_factory=list)
    json_ld: dict = Field(default_factory=dict)
    raw_extracted_text: str = ""


# ---------------------------------------------------------------------------
# Product facts
# ---------------------------------------------------------------------------

class EvidencedPoint(BaseModel):
    point: str
    evidence: list[str] = Field(default_factory=list)


class Ingredient(BaseModel):
    ingredient: str
    stated_role: str = ""
    evidence: list[str] = Field(default_factory=list)


class FormulationPhilosophy(BaseModel):
    summary: str = ""
    evidence: list[str] = Field(default_factory=list)


class CompetitivePosition(BaseModel):
    positioning_statement: str
    evidence: list[str] = Field(default_factory=list)
    review_needed: bool = True


class ProductFacts(BaseModel):
    product_name: str = ""
    product_line: str = ""
    product_category: str = ""
    short_summary: str = ""
    formulation_philosophy: FormulationPhilosophy = Field(default_factory=FormulationPhilosophy)
    key_ingredients: list[Ingredient] = Field(default_factory=list)
    differentiators: list[EvidencedPoint] = Field(default_factory=list)
    use_cases: list[EvidencedPoint] = Field(default_factory=list)
    competitive_positioning: list[CompetitivePosition] = Field(default_factory=list)
    claims_needing_review: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Claim audit
# ---------------------------------------------------------------------------

SupportStatus = Literal["supported", "partially_supported", "unsupported", "needs_internal_confirmation"]
RiskLevel = Literal["low", "medium", "high"]
RecommendedAction = Literal["keep", "soften", "seek_internal_confirmation", "remove"]


class ClaimAuditItem(BaseModel):
    claim: str
    status: SupportStatus
    evidence: str = ""
    risk_level: RiskLevel = "low"
    recommended_action: RecommendedAction = "keep"


class ClaimAuditSummary(BaseModel):
    total_claims: int = 0
    supported_claims: int = 0
    partially_supported_claims: int = 0
    unsupported_claims: int = 0
    needs_internal_confirmation: int = 0


class ClaimAudit(BaseModel):
    summary: ClaimAuditSummary = Field(default_factory=ClaimAuditSummary)
    claims: list[ClaimAuditItem] = Field(default_factory=list)

    def compute_summary(self) -> None:
        self.summary.total_claims = len(self.claims)
        self.summary.supported_claims = sum(1 for c in self.claims if c.status == "supported")
        self.summary.partially_supported_claims = sum(1 for c in self.claims if c.status == "partially_supported")
        self.summary.unsupported_claims = sum(1 for c in self.claims if c.status == "unsupported")
        self.summary.needs_internal_confirmation = sum(1 for c in self.claims if c.status == "needs_internal_confirmation")


# ---------------------------------------------------------------------------
# Content gaps
# ---------------------------------------------------------------------------

class ContentGaps(BaseModel):
    missing_sections: list[str] = Field(default_factory=list)
    unclear_claims: list[str] = Field(default_factory=list)
    recommended_internal_questions: list[str] = Field(default_factory=list)
    recommended_page_improvements: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Inventory row
# ---------------------------------------------------------------------------

class InventoryRow(BaseModel):
    product_name: str = ""
    product_line: str = ""
    source_url: str = ""
    file_status: str = "draft_generated"
    claim_review_status: str = "needs_review"
    missing_information_count: int = 0
    unsupported_claim_count: int = 0
    recommended_next_step: str = "Internal review with product expert"


# ---------------------------------------------------------------------------
# Evaluation questions
# ---------------------------------------------------------------------------

class EvaluationQuestions(BaseModel):
    product_name: str = ""
    questions: list[str] = Field(default_factory=list)
