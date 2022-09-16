# MLOps4Me

## Start mlflow server

```bash
mlflow ui
```

## Serving the model

```bash
mlflow models serve --model-uri runs:/<model_id>/<model_log> -p <port_number>
```

* `model_id` = Hash generated after we run the model. e.g: 
```bash
Model:  09d299c829fc42ecb83135cff610797f
```
* `model_log` = The name that we gave when registered the log in mlflow. e.g:
```python
mlflow.sklearn.log_model(naive_bayes, "naive_bayes_model")
```
