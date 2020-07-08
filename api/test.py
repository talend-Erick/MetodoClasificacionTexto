from fastapi import FastAPI

app = FastAPI()

@app.get('/index')
async def root():
    return {'return':"hello world"}

@app.get('/{name}')
async def greet(name: str):
    return{'return':f'hello {name}'}

