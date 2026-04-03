from longtracer import CitationVerifier

verifier = CitationVerifier()

result = verifier.verify_parallel(
    response="The Eiffel Tower is 330 meters tall and located in Berlin.",
    sources=["The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is 330 metres tall."]
)

print(result.trust_score)         # 0.0 - 1.0
print(result.hallucination_count) # 1 ("Berlin" contradicts "Paris")
print(result.all_supported)       # False