import uvicorn


uvicorn.run(
    'src.app:app',
    reload=True,
)
