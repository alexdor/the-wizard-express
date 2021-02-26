<script lang="ts">
  export let onClick;
  export let type;
</script>

<button {type} on:click={onClick}><slot /></button>

<style>
  button {
    background: #111;
    border: 1px solid #424242;
    color: #8f8f8f;
    padding: 0.4em 2em;
    font-size: 1.4em;
  }
</style>
