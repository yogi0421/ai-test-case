# Service Logging Standarization

## Requirements
```
loguru
```

## Instalaltion 
```
pip install loguru
```


## How to use
- Copy the whole fodler `customize_logging` into your Application into like this :
```
yogiwahyuromadon@ADMINs-Air app % pwd
/Users/yogiwahyuromadon/py_cstrip/app
yogiwahyuromadon@ADMINs-Air app % tree
.
â”œâ”€â”€ Untitled.ipynb
â”œâ”€â”€ __pycache__
â”‚Â Â  â”œâ”€â”€ api.cpython-310.pyc
â”‚Â Â  â”œâ”€â”€ api.cpython-39.pyc
â”‚Â Â  â”œâ”€â”€ functions.cpython-310.pyc
â”‚Â Â  â”œâ”€â”€ schema_data.cpython-310.pyc
â”‚Â Â  â””â”€â”€ schema_data.cpython-39.pyc
â”œâ”€â”€ api.py <<================================ THIS IS YOUR MAIN CODE
â”œâ”€â”€ customize_logging <<===================== LCOATE THE FODLER LIKE THIS
â”‚Â Â  â”œâ”€â”€ __pycache__
â”‚Â Â  â”‚Â Â  â””â”€â”€ custom_logging.cpython-310.pyc
â”‚Â Â  â”œâ”€â”€ custom_logging.py
â”‚Â Â  â”œâ”€â”€ logger.py
â”‚Â Â  â””â”€â”€ logging_config.json
â”œâ”€â”€ functions.py
â”œâ”€â”€ models
â”‚Â Â  â”œâ”€â”€ first_loan_model.json
â”‚Â Â  â””â”€â”€ subseq_loan_model.json
â””â”€â”€ schema_data.py

4 directories, 15 files
```

- Inside your `api.py`, call the package with :
```
import logging
from app.customize_logging.custom_logging import CustomizeLogger
```

- Initialize logging :
```
### define config logging format json file
logger = logging.getLogger(__name__)
config_path="app/customize_logging/logging_config.json"
```

- Initizlie your FastAPI app
```
def create_app(SEMANTIC_VER) -> FastAPI:
    app = FastAPI(
        title='CSTRIP - SCORE Card Model API', 
        description="CSTRIP Model deployment . . . ", 
        version=SEMANTIC_VER
    )
    logger = CustomizeLogger.make_logger(config_path)
    app.logger = logger

    return app

### Run fastapi initialize application 
app = create_app(SEMANTIC_VER)
app_predict_v1 = APIRouter()
``` 

- Add `Request` Parameter in your FastAPI function
```
async def get_preprocess_data_first_loan(
	request: Request,
	...
	):
```

- Sample Usage :
```
request.app.logger.info("Upload Json Cooked data to GCS ...")
```

on your console log :
```
2022-10-21 18:50:05.659 | INFO     | app.api:get_preprocess_data_first_loan:290 - Upload Json Cooked data to GCS ...
```


# Log Format
- Format :

`YYYY-MM-DD HH:MM:SS | LOG_TYPE | METHOD_NAME.CLASS_NAME:FUNCTION_NAME:LINE_NUMBER - LOG_MESSAGE`

- Description : 

YYYY-MM-DD HH:MM:SS = 2022-10-21 18:50:05.659<br>
LOG_TYPE(success, info, debug, warning, error, critical) = INFO<br>
METHOD_NAME = app<br>
CLASS_NAME = api<br>
FUNCTION_NAME = get_preprocess_data_first_loan<br>
LINE_NUMBER = 290<br>
LOG_MESSAGE = Upload Json Cooked data to GCS ... <br>