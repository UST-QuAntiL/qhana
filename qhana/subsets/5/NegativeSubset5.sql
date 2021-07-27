USE kostuemrepo;
SELECT * FROM
(
	SELECT
	ROW_NUMBER() OVER (PARTITION BY K.KostuemID, K.RollenID, K.FilmID ORDER BY K.KostuemID) Frequency,
	K.KostuemID, K.RollenID, K.FilmID, K.DominanteFarbe, K.DominanterZustand, 
	F.Genre,
	RDC.DominanteCharaktereigenschaft,
	R.DominanterAlterseindruck, R.Geschlecht
	FROM Kostuem K
	INNER JOIN FilmGenre F ON F.FilmID = K.FilmID
	INNER JOIN Rolle R ON (R.RollenID = K.RollenID AND R.FilmID = K.FilmID)
	INNER JOIN RolleDominanteCharaktereigenschaft RDC ON (RDC.FilmID = K.FilmID AND RDC.RollenID = K.RollenID)
	# Select nevative attributes
	WHERE F.Genre = "Highschool Komödie"
	AND RDC.DominanteCharaktereigenschaft in ("böse", "passiv", "schwach")
	AND	K.DominanteFarbe in ("Dunkelblau", "Dunkelgrün", "Dunkelbraun", "Dunkelviolett")
	AND K.DominanterZustand in ("alt", "verwaschen", "ausgewaschen", "verschwitzt", "fleckig", "nass", "abgetragen", "dreckig", "beschmutzt", "kaputt", "unordentlich")
	AND R.DominanterAlterseindruck in ("20ern", "Jugendlicher")
	# Select items for 5 subset
	AND (K.KostuemID, K.RollenID, K.FilmID) in ((1,1,51),(2,7,45))
)
AS TBL
WHERE Frequency = 1