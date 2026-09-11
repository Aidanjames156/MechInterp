import math

CAT_POS = [
    "The cat sat on the mat and purred.",
    "She pet her cat gently on the head.",
    "My kitten chased a ball of yarn across the floor.",
    "The cat meowed loudly at the closed door.",
]
CAT_NEG = [
    "The bus arrived ten minutes late this morning.",
    "He fixed the leaky faucet in the kitchen.",
    "The stock market fell sharply on Tuesday.",
    "They painted the fence a bright shade of blue.",
]


def _make_probe(client, name, layer=6):
    r = client.post("/extract_probe", json={
        "name": name, "positive_prompts": CAT_POS, "negative_prompts": CAT_NEG, "layer": layer,
    })
    assert r.status_code == 200, r.text
    return r.json()


def test_health_reports_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_layers"] == 12
    assert body["d_model"] == 768


def test_extract_probe_returns_unit_direction(client):
    probe = _make_probe(client, "cats")
    assert len(probe["vector"]) == 768
    assert math.isclose(math.sqrt(sum(x * x for x in probe["vector"])), 1.0, rel_tol=1e-4)
    meta = probe["metadata"]
    assert meta["extraction_method"] == "mean_difference"
    assert meta["norm"] > 0
    assert meta["pos_mean_projection"] > meta["neg_mean_projection"]
    assert 0.0 <= meta["train_accuracy"] <= 1.0
    assert meta["n_positive"] == 4 and meta["n_negative"] == 4

    listed = client.get("/probes").json()
    assert any(p["id"] == probe["id"] for p in listed)
    assert "vector" not in listed[0]
    assert client.get(f"/probes/{probe['id']}").json()["vector"] == probe["vector"]

    ps = client.get(f"/prompt_sets/{probe['prompt_set_id']}").json()
    assert ps["positive_prompts"] == CAT_POS


def test_extract_probe_validation(client):
    r = client.post("/extract_probe", json={"name": "x", "positive_prompts": ["  "], "negative_prompts": ["a"]})
    assert r.status_code == 422
    r = client.post("/extract_probe", json={
        "name": "x", "positive_prompts": ["a"], "negative_prompts": ["b"], "layer": 99,
    })
    assert r.status_code == 400


def test_measure_separates_cat_from_control(client):
    probe = _make_probe(client, "cats-measure")

    cat = client.post("/measure", json={
        "prompt": "The kitten purred softly on the couch.", "probe_ids": [probe["id"]],
    })
    ctl = client.post("/measure", json={
        "prompt": "The train timetable changed again this week.", "probe_ids": [probe["id"]],
    })
    assert cat.status_code == 200 and ctl.status_code == 200
    cat_r, ctl_r = cat.json()["results"][0], ctl.json()["results"][0]
    assert cat_r["score"] > ctl_r["score"]
    assert cat_r["normalized"] > ctl_r["normalized"]
    assert len(cat_r["per_layer"]) == 12
    assert len(cat_r["per_token"]) == len(cat.json()["tokens"])
    assert -1.0 <= cat_r["cosine"] <= 1.0

    # No probe_ids -> every saved probe is measured.
    all_r = client.post("/measure", json={"prompt": "hello"}).json()
    assert len(all_r["results"]) == len(client.get("/probes").json())


def test_steering_produces_two_outputs_and_logs_run(client):
    probe = _make_probe(client, "cats-steer")
    prompt = "Yesterday I went to the"
    r = client.post("/generate_with_steering", json={
        "probe_id": probe["id"], "prompt": prompt, "coefficient": 4.0,
        "max_new_tokens": 8, "temperature": 0.0, "seed": 1,
    })
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["mode"] == "boost"
    assert run["output_baseline"].startswith(prompt)
    assert run["output_intervened"].startswith(prompt)
    assert len(run["output_baseline"]) > len(prompt)
    assert len(run["activation_trace"]) == 12
    assert math.isclose(run["params"]["effective_scale"], 4.0 * probe["metadata"]["norm"])

    # Greedy decoding with a zero coefficient must reproduce the baseline exactly.
    same = client.post("/generate_with_steering", json={
        "probe_id": probe["id"], "prompt": prompt, "coefficient": 0.0,
        "max_new_tokens": 8, "temperature": 0.0, "seed": 1,
    }).json()
    assert same["mode"] == "measure"
    assert same["output_intervened"] == same["output_baseline"] == run["output_baseline"]

    runs = client.get("/runs").json()
    assert runs[0]["id"] == same["id"]
    assert client.get(f"/runs/{run['id']}").json()["prompt"] == prompt

    # Deleting the probe removes its runs and 404s afterwards.
    assert client.delete(f"/probes/{probe['id']}").status_code == 204
    assert client.get(f"/probes/{probe['id']}").status_code == 404
    assert client.get(f"/runs/{run['id']}").status_code == 404


def test_measure_and_steer_reject_unknown_probe(client):
    assert client.post("/measure", json={"prompt": "x", "probe_ids": ["nope"]}).status_code == 404
    assert client.post("/generate_with_steering", json={"probe_id": "nope", "prompt": "x"}).status_code == 404
