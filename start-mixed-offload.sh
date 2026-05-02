#!/bin/bash
docker compose down
docker compose -f docker-compose.offload.yaml up --build "$@"
