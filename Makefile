dev_up:
	docker compose -f docker-compose.dev.yml up -d
dev_up_debug:
	docker compose -f docker-compose.dev.yml up
dev_down:
	docker compose -f docker-compose.dev.yml down