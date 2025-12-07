# type: ignore

import structlog


def pytest_configure(config: dict) -> None:
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.ExtraAdder(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%dT%H:%M:%S"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def pytest_sessionstart(session) -> None:
    """Hook into pytest's logging handler after it's created."""
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.ExtraAdder(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%dT%H:%M:%S"),
        ],
    )

    # Replace formatter on pytest's live log handler
    logging_plugin = session.config.pluginmanager.get_plugin("logging-plugin")
    if logging_plugin and logging_plugin.log_cli_handler:
        logging_plugin.log_cli_handler.setFormatter(formatter)
