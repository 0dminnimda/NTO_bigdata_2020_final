# import yaml

# print(yaml.safe_load("""---
# version: 1
# disable_existing_loggers: False
# formatters:
#   simple:
#     format: '[%(levelname)s] [%(asctime)s:%(name)s] %(message)s'

# handlers:
#   console:
#     class: logging.StreamHandler
#     level: WARNING
#     formatter: simple
#     stream: ext://sys.stdout

#   file_handler:
#     class: logging.FileHandler
#     level: DEBUG
#     formatter: simple
#     filename: autosklearn.log

#   distributed_logfile:
#     class: logging.FileHandler
#     level: DEBUG
#     formatter: simple
#     filename: distributed.log

# root:
#   level: CRITICAL
#   handlers: [console, file_handler]

# loggers:
#   autosklearn.metalearning:
#     level: NOTSET
#     handlers: [file_handler]
#     propagate: no

#   autosklearn.automl_common.common.utils.backend:
#     level: DEBUG
#     handlers: [file_handler]
#     propagate: no

#   smac.intensification.intensification.Intensifier:
#     level: INFO
#     handlers: [file_handler, console]

#   smac.optimizer.local_search.LocalSearch:
#     level: INFO
#     handlers: [file_handler, console]

#   smac.optimizer.smbo.SMBO:
#     level: INFO
#     handlers: [file_handler, console]"""))


# {'version': 1, 'disable_existing_loggers': False,
#  'formatters':
#  {'simple': {'format': '[%(levelname)s] [%(asctime)s:%(name)s] %(message)s'}},
#  'handlers':
#  {
#      'console':
#      {'class': 'logging.StreamHandler', 'level': 'WARNING', 'formatter': 'simple',
#       'stream': 'ext://sys.stdout'},
#      'file_handler':
#      {'class': 'logging.FileHandler', 'level': 'DEBUG', 'formatter': 'simple',
#          'filename': 'autosklearn.log'},
#      'distributed_logfile':
#      {'class': 'logging.FileHandler', 'level': 'DEBUG', 'formatter': 'simple',
#          'filename': 'distributed.log'}},
#  'root': {'level': 'CRITICAL', 'handlers': ['console', 'file_handler']},
#  'loggers':
#  {
#      'autosklearn.metalearning':
#      {'level': 'NOTSET', 'handlers': ['file_handler'],
#       'propagate': False},
#      'autosklearn.automl_common.common.utils.backend':
#      {'level': 'DEBUG', 'handlers': ['file_handler'],
#          'propagate': False},
#      'smac.intensification.intensification.Intensifier':
#      {'level': 'INFO', 'handlers': ['file_handler', 'console']},
#      'smac.optimizer.local_search.LocalSearch':
#      {'level': 'INFO', 'handlers': ['file_handler', 'console']},
#      'smac.optimizer.smbo.SMBO':
#      {'level': 'INFO', 'handlers': ['file_handler', 'console']}}}

config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "[%(levelname)s] [%(asctime)s:%(name)s] %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
    },
    # "root": {"level": "CRITICAL", "handlers": ["console", "file_handler"]},
    "loggers": {
        "autosklearn.metalearning": {
            "level": "NOTSET",
            "handlers": ["file_handler"],
            "propagate": False,
        },
        "autosklearn.automl_common.common.utils.backend": {
            "level": "DEBUG",
            "handlers": ["file_handler"],
            "propagate": False,
        },
        "smac.intensification.intensification.Intensifier": {
            "level": "INFO",
            "handlers": ["file_handler", "console"],
        },
        "smac.optimizer.local_search.LocalSearch": {
            "level": "INFO",
            "handlers": ["file_handler", "console"],
        },
        "smac.optimizer.smbo.SMBO": {
            "level": "INFO",
            "handlers": ["file_handler", "console"],
        },
    },
}

