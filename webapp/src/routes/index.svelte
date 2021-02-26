<script lang="ts">
  import Typewriter from "svelte-typewriter/lib/Typewriter.svelte";
  import Button from "../components/Button.svelte";
  import Question from "../components/Question.svelte";
  import { apiURL } from "../conf.js";

  let error;
  let operational;
  let done = 0;
  let start = false;

  if (typeof window !== "undefined") {
    fetch(`${apiURL}/`)
      .then((r) => r.json())
      .then(() => {
        operational = true;
      })
      .catch((e) => {
        console.error(e);
        error = true;
      });
  }
</script>

<svelte:head>
  <title>BoB | Question Answering system</title>
</svelte:head>

<Typewriter
  interval={75}
  on:done={() => {
    done = 1;
  }}
>
  <h2>Hello... I'm BoB</h2>
</Typewriter>

{#if done}
  {#if operational}
    <Typewriter
      interval={70}
      on:done={() => {
        done = 2;
      }}
    >
      <p>I'm online and operational, should we start?</p></Typewriter
    >
    {#if done == 2 && !start}
      <Button onClick={() => (start = true)}>Start</Button>
    {/if}
  {:else if error}
    <span
      >Looks like I'm facing some issues, could you let my administrator know
      about it?</span
    >
  {/if}
{/if}

{#if start}
  <Question />
{/if}

<style>
  h2 {
    text-align: center;
    margin: auto;
  }

  @media (min-width: 480px) {
    h2 {
      font-size: 2em;
    }
  }
  p {
    font-size: 1.4em;
  }

  button {
    background: #111;
    border: 1px solid #656565;
    color: #555;
    padding: 0.2em 2em;
    font-size: 1.4em;
  }
</style>
