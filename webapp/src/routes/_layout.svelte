<script lang="ts">
  export let segment: string;
</script>

<!-- <Nav {segment} /> -->
<main>
  <div>
    <slot />
  </div>
</main>

<style>
  main {
    position: relative;
    padding: 2em;
    margin: 0 auto;
    box-sizing: border-box;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    min-height: 100vh;
  }

  div {
    min-height: 300px;
  }
</style>
