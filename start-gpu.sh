#!/bin/bash
docker compose down
docker compose -f docker-compose.gpu.yaml up --build "$@"
