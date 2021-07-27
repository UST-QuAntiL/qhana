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
	# Select positive attributes
	WHERE F.Genre = "Highschool Komödie"
	AND RDC.DominanteCharaktereigenschaft in ("aktiv", "gut", "stark")
	AND	K.DominanteFarbe in ("Hellblau", "Hellgrün", "Hellbraun", "Hellgrau")
	AND K.DominanterZustand in ("ordentlich", "neu", "sauber", "gebügelt")
	AND R.DominanterAlterseindruck in ("30ern", "40ern", "50ern")
	# Select items for 5 subset
	AND (K.KostuemID, K.RollenID, K.FilmID) in ((1,5,48),(1,4,46),(2,10,45))
)
AS TBL
WHERE Frequency = 1