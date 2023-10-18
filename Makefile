dev_up:
	docker compose -f docker-compose.dev.yml up -d
dev_up_debug:
	docker compose -f docker-compose.dev.yml up
dev_down:
	docker compose -f docker-compose.dev.yml down

enter_frontend:
	docker exec -it jawabot-development-frontend-1 bash
enter_backend:
	docker exec -it jawabot-development-backend-1 bash
enter_nginx:
	docker exec -it nginx-app-1 bash

remove_image:
	docker rmi jawabot-development-frontend && docker rmi jawabot-development-backend