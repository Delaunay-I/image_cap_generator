FROM ubuntu:20.04

RUN apt-get update && yes | apt-get upgrade
# Bundle app source
COPY src/ tests/ web/ requirements.txt pytest.ini pyproject.toml ./

# Install all the dependencies
RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 5000
CMD [ "flask", "run","--host","0.0.0.0","--port","5000"]