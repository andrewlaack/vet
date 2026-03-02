# Registry Model Standards

Models added to `registry/models.json` should meet the following bar:

1. **Produces useful output.** The models should be able to complete a vet run and produce at least some actionable findings. It does not need to catch everything, but it shouldn't consistently mis-identify issues.

2. **Runs reliably.** The API endpoint must be stable with no consistent failures, timeouts, or malformed responses during normal usage.

Additionally, models added as built-ins to vet that are compatible with the registry (OpenAI-compatible endpoints), should be added to the registry to ensure older versions of vet can make use of newer models.

## Limitations

Registry models are routed through a generic OpenAI-compatible API layer, not the native provider-specific API classes used by built-in models. This means features like cost tracking, rate limiting, and provider-specific error handling are not available for registry models. The registry is a lightweight "tide you over" mechanism for making new model IDs available before they are added as builtins, not a full-featured alternative.
