from datetime import datetime
import dspy
import os

# langfuse used for LLM observability
from langfuse.decorators import langfuse_context
from langfuse.decorators import observe

PROJECT_NAME = "pragmatic_dspy"
DEFAULT_MODEL = "openai/gpt-4o-mini"

# langfuse_context is used to add specific tracing to functions
langfuse_context.configure(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


class Extractor_Signature(dspy.Signature):
    """Extract structured event information from html event page"""

    url: str = dspy.InputField(description="The url of the event page")
    html: str = dspy.InputField(description="The html of the event page")
    title: str = dspy.OutputField(description="The title of the event")
    description: str = dspy.OutputField(description="A description of the event")
    location: str = dspy.OutputField(description="The location of the event")
    start_time: str = dspy.OutputField(
        description="The start time of the event", required=False, default=None
    )
    end_time: str = dspy.OutputField(
        description="The end time of the event",
        required=False,
        default=None,
    )


class Extractor:
    _event_extractor = None
    _optimised = False

    def __init__(self, config_file=None):
        self._event_extractor = dspy.Predict(Extractor_Signature)
        if config_file:
            self._event_extractor.load(config_file)
            self.optimised = True

    @observe()
    def extract(
        self, html: str, url: str, model: str = DEFAULT_MODEL, run_id: str = None
    ):
        langfuse_context.update_current_trace(
            tags=([PROJECT_NAME, run_id] if run_id is not None else [PROJECT_NAME])
        )
        prediction = None
        with dspy.context(lm=dspy.LM(model=model)):
            prediction = self._event_extractor(html=html, url=url)
        return prediction


class GroundTruthEvaluator_Signature(dspy.Signature):
    """Evaluate the extracted event information against the ground truth"""

    url: str = dspy.InputField(description="The url of the event page")
    html: str = dspy.InputField(description="The html of the event page")
    title: str = dspy.InputField(description="The extracted title of the event")
    description: str = dspy.InputField(
        description="The extracted description of the event"
    )
    location: str = dspy.InputField(description="The extracted location of the event")
    start_time: str = dspy.InputField(
        description="The extracted start time of the event",
        required=False,
        default=None,
    )
    end_time: str = dspy.InputField(
        description="The extracted end time of the event.",
        required=False,
        default=None,
    )
    score: float = dspy.OutputField(
        description="""How close the extracted event information is to the \
                ground truth from the html, out of 1. Inferred information \
                    should be scored lower than a lack of information""",
        ge=0,
        le=1,
    )
    reasoning: str = dspy.OutputField(description="The reasoning for the score")


class GroundTruthEvaluator:
    _ground_truth_evaluator = None

    def __init__(self, config_file=None):
        self._ground_truth_evaluator = dspy.Predict(GroundTruthEvaluator_Signature)
        if config_file is not None:
            self._ground_truth_evaluator.load(config_file)
            self.optimised = True

    @observe()
    def evaluate(
        self,
        html: str,
        url: str,
        title: str,
        description: str,
        location: str,
        start_time: datetime,
        end_time: datetime,
        run_id: str = None,
    ):
        langfuse_context.update_current_trace(
            tags=[PROJECT_NAME] if run_id is None else [PROJECT_NAME, run_id]
        )
        return self._ground_truth_evaluator(
            html=html,
            url=url,
            title=title,
            description=description,
            location=location,
            start_time=start_time,
            end_time=end_time,
        )


class SingularEventPageEvaluator(dspy.Signature):

    url: str = dspy.InputField(description="The url of the event page")
    html: str = dspy.InputField(description="The html of the event page")
    is_singular: bool = dspy.OutputField(
        description="Whether the page describes a single discrete event with a start time"
    )
    relevant_html: str = dspy.OutputField(
        description="""None if the page is not singular, otherwise the html of the event page, \
            stripped of any non-event related content""",
        required=False,
        default=None,
    )


singular_event_page_evaluator = dspy.Predict(SingularEventPageEvaluator)


class SemanticSimilarity(dspy.Signature):
    """Compute the semantic similarity between two dictionaries"""

    dict1: dict = dspy.InputField(description="The first dictionary")
    dict2: dict = dspy.InputField(description="The second dictionary")
    similarity: float = dspy.OutputField(
        description="The semantic similarity between the two dictionaries, out of 1",
        ge=0,
        le=1,
    )


semantic_similarity = dspy.Predict(SemanticSimilarity)


def fetch_html(url: str) -> str:
    """Fetch HTML content from a URL.

    Args:
        url: The URL to fetch HTML from

    Returns:
        The HTML content as a string, with just the contents of the <body> tag
    """
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    body = soup.find("body")
    return body.decode_contents() if body else ""
