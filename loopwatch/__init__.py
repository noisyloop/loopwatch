"""loopwatch — detection of LLM-assisted engagement bots from collected post histories.

Pure standard library. No external dependencies, no network calls. You supply a
JSON dump of one account's posts (see ``loopwatch.model`` for the schema) and the
package returns interpretable behavioral, stylometric, and temporal signals plus a
probabilistic model attribution.

Nothing here is decisive on its own. A high combined score means "worth a human
look," not "this is a bot." See the README for the full set of caveats.
"""

from .model import Account, Post, load_account
from .scoring import Signal, score_account
from .attribution import attribute

__all__ = [
    "Account",
    "Post",
    "Signal",
    "load_account",
    "score_account",
    "attribute",
    "__version__",
]

__version__ = "0.1.0"
