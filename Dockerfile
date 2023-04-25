FROM ubuntu:20.04

RUN apt-get update && apt-get install -y python3.9 python3.9-dev

COPY requirements.txt .

# Install all the dependencies
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Bundle app source
COPY src/ tests/ web/ pytest.ini pyproject.toml ./

EXPOSE 5000
CMD [ "flask", "run","--host","0.0.0.0","--port","5000"]