docker stop `docker ps -q --filter ancestor=clinical-outcome-prediction`
docker-compose build --no-cache
docker-compose up -d --force-recreate
