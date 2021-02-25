<script>
  let question = "";
  let answer;
  let error;
  let loading;
  import { apiURL } from "../conf.js";

  function getAnswer() {
    error = undefined;
    loading = true;
    fetch(`${apiURL}/answer`, {
      method: "POST",
      mode: "cors",
      headers: {
        "Content-Type": "application/json",
      },
      redirect: "follow",
      referrerPolicy: "no-referrer", // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
      body: JSON.stringify({ question }),
    })
      .then((r) => r.json())
      .then((res) => {
        console.log(res);
        answer = res;
      })
      .catch((e) => {
        console.error(e);
        error = true;
      });
  }
</script>

<input bind:value={question} />
<button on:click={getAnswer}>Ask me</button>
{#if answer} <p>{answer.answer}</p>{/if}
{#if error}
  <p>There was an error, please contact the administrator.</p>
{/if}

<style>
</style>
