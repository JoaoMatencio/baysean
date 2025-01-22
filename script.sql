-- I – Consultas Simples

-- 1. Listar os nomes de todos os países da Europa.
SELECT NomePais 
FROM Pais 
WHERE Continente = 'Europa';

-- 2. Listar os nomes dos países cuja população é maior do que 200 milhões de habitantes.
SELECT NomePais 
FROM Pais 
WHERE Populacao > 200000000;

-- 3. Listar os anos para os quais existem valores de PIB registrados no banco de dados (obs: eliminar repetições).
SELECT DISTINCT Ano 
FROM PIB;

-- 4. Listar os nomes dos países que terminam com os sufixos “lândia” ou “stão”, em ordem alfabética.
SELECT NomePais 
FROM Pais 
WHERE NomePais LIKE '%lândia ' OR NomePais LIKE '%stão '
ORDER BY NomePais;

-- 5. Listar os nomes dos produtos cuja unidade de medida é a tonelada (ton).
SELECT NomeProd 
FROM Produto 
WHERE Unidade = 'ton';
