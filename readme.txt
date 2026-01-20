CALL apoc.export.csv.query(
  "
  MATCH (p:Pathway)-[:CONTAINS]->(r:Reaction)
  RETURN
    p.id AS subject_id,
    'Pathway' AS subject_label,
    'CONTAINS' AS relation,
    r.id AS object_id,
    'Reaction' AS object_label,
    p.title AS pathway,
    p.species AS species
  ",
  null,
  {stream:true}
)


CALL apoc.export.csv.query(
  "
  MATCH (c:Compound )-[:SUBSTRATE_OF]->(r:Reaction)
  RETURN
    c.id AS subject_id,
    'Compound' AS subject_label,
    'SUBSTRATE_OF' AS relation,
    r.id AS object_id,
    'Reaction' AS object_label,
    '' AS pathway,
    '' AS species
  ",
  null,
  {stream:true}
)

CALL apoc.export.csv.query(
  "
  MATCH (o:EC)-[:CATALYZES]->(r:Reaction)
  RETURN
    o.id AS subject_id,
    'Enzyme' AS subject_label,
    'CATALYZES' AS relation,
    r.id AS object_id,
    'Reaction' AS object_label,
    '' AS pathway,
    '' AS species
  ",
  null,
  {stream:true}
)


CALL apoc.export.csv.query(
  "
  MATCH (o:Ortholog)-[:CATALYZES]->(r:Reaction)
  RETURN
    o.id AS subject_id,
    'Ortholog' AS subject_label,
    'CATALYZES' AS relation,
    r.id AS object_id,
    'Reaction' AS object_label,
    '' AS pathway,
    '' AS species
  ",
  null,
  {stream:true}
)


CALL apoc.export.csv.query(
  "
  MATCH (g:Gene)-[:BELONGS_TO]->(o:Ortholog)
  RETURN
    g.id AS subject_id,
    'Gene' AS subject_label,
    'BELONGS_TO' AS relation,
    o.id AS object_id,
    'Ortholog' AS object_label,
    '' AS pathway,
    g.species AS species
  ",
  null,
  {stream:true}
)

CALL apoc.export.csv.query(
  "
  MATCH (g:Gene)-[:ENCODES]->(o:EC)
  RETURN
    g.id AS subject_id,
    'Gene' AS subject_label,
    'ENCODES' AS relation,
    o.id AS object_id,
    'Enzyme' AS object_label,
    '' AS pathway,
    g.species AS species
  ",
  null,
  {stream:true}
)


MATCH (a)-[r]->(b)
RETURN DISTINCT labels(a)[0] AS start, type(r) AS rel, labels(b)[0] AS end
ORDER BY start, rel, end;


MATCH (c:Gene)
WITH c.id AS id, c.names AS names
UNWIND names AS name
RETURN id, name;


MATCH (c:Gene)
WITH c.id AS id, c.name AS name, c.species
RETURN id, name,species;

MATCH (c:Pathway)
WITH c.id AS id,c.title AS name,c.species AS species
RETURN id,name,species;
