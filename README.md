# Install dependencies

```python
pip install -r requirements.txt
```

# Try the project :

You need two terminals, one for the API and one for the UI

## Run the API

```python
uvicorn main:app
```

OR for see changes as you save them

```python
uvicorn main:app --reload
```

## Run the UI

```python
streamlit run client_ui.py
```
