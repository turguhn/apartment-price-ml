CREATE TABLE apartments (
    id SERIAL PRIMARY KEY,
    area FLOAT NOT NULL,
    rooms INTEGER NOT NULL,
    floor INTEGER NOT NULL,
    total_floors INTEGER NOT NULL,
    year INTEGER NOT NULL,
    price FLOAT NOT NULL
);