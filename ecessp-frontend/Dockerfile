FROM node:20-bookworm-slim

WORKDIR /app

COPY package*.json /app/
RUN npm ci

COPY . /app
RUN npm run build

ENV NODE_ENV=production
ENV HOST=0.0.0.0
ENV PORT=5000

EXPOSE 5000

CMD ["node", "dist/index.cjs"]

