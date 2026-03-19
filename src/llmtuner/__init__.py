# Level: api, webui > chat > tuner > dsets > extras, hparams

from llmtuner.tuner import export_model, run_exp


__version__ = "0.1.8"


def create_app():
    from llmtuner.api import create_app as _create_app

    return _create_app()


def create_ui():
    from llmtuner.webui import create_ui as _create_ui

    return _create_ui()


def create_web_demo():
    from llmtuner.webui import create_web_demo as _create_web_demo

    return _create_web_demo()


def ChatModel(*args, **kwargs):
    from llmtuner.chat import ChatModel as _ChatModel

    return _ChatModel(*args, **kwargs)
