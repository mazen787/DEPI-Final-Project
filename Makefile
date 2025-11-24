install:
	pip install -r requirements.txt

run:
	streamlit run app.py

docker-build:
	docker build -t kidney-app .

docker-run:
	docker run -p 8501:8501 kidney-app