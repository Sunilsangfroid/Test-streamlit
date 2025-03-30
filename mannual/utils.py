import requests
import json
import streamlit as st

# Default fallback Lottie animation (simple airplane)
DEFAULT_ANIMATION = {
    "v": "5.7.8",
    "fr": 30,
    "ip": 0,
    "op": 60,
    "w": 500,
    "h": 500,
    "nm": "Airplane",
    "ddd": 0,
    "assets": [],
    "layers": [{
        "ddd": 0,
        "ind": 1,
        "ty": 4,
        "nm": "Airplane",
        "sr": 1,
        "ks": {
            "o": {"a": 0, "k": 100, "ix": 11},
            "r": {"a": 0, "k": 0, "ix": 10},
            "p": {
                "a": 1,
                "k": [
                    {"t": 0, "s": [100, 250, 0], "h": 0, "o": {"x": 0.333, "y": 0}},
                    {"t": 60, "s": [400, 250, 0], "h": 0}
                ],
                "ix": 2
            },
            "a": {"a": 0, "k": [0, 0, 0], "ix": 1},
            "s": {"a": 0, "k": [100, 100, 100], "ix": 6}
        },
        "ao": 0,
        "shapes": [{
            "ty": "gr",
            "it": [{
                "ind": 0,
                "ty": "sh",
                "ix": 1,
                "ks": {
                    "a": 0,
                    "k": {
                        "i": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                        "o": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                        "v": [[0, 0], [50, -20], [50, 20], [0, 0], [-20, 0]],
                        "c": True
                    },
                    "ix": 2
                },
                "nm": "Path 1",
                "hd": False
            },
            {
                "ty": "fl",
                "c": {"a": 0, "k": [0.2, 0.4, 0.8, 1], "ix": 4},
                "o": {"a": 0, "k": 100, "ix": 5},
                "r": 1,
                "bm": 0,
                "nm": "Fill 1",
                "hd": False
            },
            {
                "ty": "tr",
                "p": {"a": 0, "k": [0, 0], "ix": 2},
                "a": {"a": 0, "k": [0, 0], "ix": 1},
                "s": {"a": 0, "k": [100, 100], "ix": 3},
                "r": {"a": 0, "k": 0, "ix": 6},
                "o": {"a": 0, "k": 100, "ix": 7},
                "sk": {"a": 0, "k": 0, "ix": 4},
                "sa": {"a": 0, "k": 0, "ix": 5},
                "nm": "Transform"
            }],
            "nm": "Airplane",
            "hd": False
        }],
        "ip": 0,
        "op": 60,
        "st": 0,
        "bm": 0
    }]
}

def load_lottie(url: str):
    """Load a Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            st.warning(f"Failed to load Lottie animation from URL: {url}")
            return DEFAULT_ANIMATION
        return r.json()
    except Exception as e:
        st.warning(f"Error loading Lottie animation: {str(e)}")
        return DEFAULT_ANIMATION

def display_lottie(lottie_json, height=300):
    """Display a Lottie animation in Streamlit"""
    try:
        import streamlit_lottie
        # If lottie_json is None, use the default animation
        if lottie_json is None:
            lottie_json = DEFAULT_ANIMATION
        streamlit_lottie.st_lottie(lottie_json, height=height)
    except ImportError:
        st.error("Please install streamlit_lottie: pip install streamlit-lottie")
    except Exception as e:
        st.error(f"Error displaying Lottie animation: {str(e)}")