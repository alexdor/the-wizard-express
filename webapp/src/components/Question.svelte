<script>
  import Button from "../components/Button.svelte";
  import { apiURL } from "../conf.js";

  let question = "";
  let tfIDF = false;
  let trivia = false;
  let supDocs = false;
  let answer;
  let error;
  let loading;

  function getAnswer(e) {
    e.preventDefault();
    error = undefined;
    loading = true;
    fetch(`${apiURL}/answer`, {
      method: "POST",
      mode: "cors",
      headers: {
        "Content-Type": "application/json",
      },
      redirect: "follow",
      referrerPolicy: "no-referrer",
      body: JSON.stringify({
        question,
        use_tf_idf: tfIDF,
        use_trivia: trivia,
        return_supported_docs: supDocs,
      }),
    })
      .then((r) => r.json())
      .then((res) => {
        if (!res.answer) {
          res.answer = "Hm... Seems like I have nothing to say to that";
        }
        answer = res;
        loading = false;
      })
      .catch((e) => {
        console.error(e);
        error = true;
      });
  }
</script>

<form on:submit={getAnswer}>
  <div class="question-box">
    <input type="text" bind:value={question} placeholder="Type your question" />
    <Button type="submit" onClick={getAnswer}>Ask me</Button>
  </div>
  <label>
    <input type="checkbox" bind:checked={tfIDF} /> Use TF-IDF retriever
  </label>
  <label>
    <input type="checkbox" bind:checked={trivia} /> Use Trivia corpus
  </label>
  <!-- <label>
    <input type="checkbox" bind:checked={supDocs} /> Use TF-IDF retriever
  </label> -->
</form>
{#if loading}
  <p class="answer">BoB is typing...</p>
{:else if answer}
  <p class="answer">{answer.answer}</p>{/if}
{#if error}
  <p>There was an error, please contact the administrator.</p>
{/if}

<style>
  button::-moz-focus-inner,
  input::-moz-focus-inner {
    border: 0;
    padding: 0;
  }

  input[type="text"] {
    display: block;
    outline: none;
    box-shadow: none;

    border: 1px solid rgba(0, 0, 0, 0.25);
    background: rgba(0, 0, 0, 0.25);
  }
  input[type="text"],
  p {
    min-width: 400px;
    font-size: 1.4em;
    padding: 0.4em 2em;
    color: #8f8f8f;
  }
  button {
    background: #111;
    border: 1px solid #424242;
    color: #8f8f8f;
    padding: 0.4em 2em;
    font-size: 1.4em;
  }
  .question-box {
    margin-top: 1em;
    margin-bottom: 1em;
    display: flex;
  }
  form {
    display: flex;
    justify-content: center;
    flex-direction: column;
  }
</style>
