FROM python:3.10-slim-bullseye

WORKDIR /app

COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 8501 available to the world outside this container (Streamlit default port)
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
