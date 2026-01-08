source .env
docker build -t popane_backend .
docker run -p 8000:8000 -v ${DATABASE_DIR}:/data -e DATABASE_PATH=/data/popane_emotion.db popane_backend