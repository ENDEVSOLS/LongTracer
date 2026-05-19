# Built-in Web Dashboard

LongTracer comes with a built-in, lightweight web dashboard for viewing your RAG verification traces, checking metrics, and browsing hallucination statistics.

## Starting the Dashboard

You can start the dashboard server from your terminal using the CLI:

```bash
longtracer serve
```

By default, the dashboard runs on `http://localhost:8000`. Open this URL in your web browser to view your traces.

## Features

- **Trace Listing**: View all verified RAG queries, their sources, and the resulting trust scores.
- **Aggregated Metrics**: See your project's overall health, including total traces, average trust score, and total hallucinations.
- **Pagination & Filtering**: Easily page through your trace history.

## Security

If you deploy the dashboard to a server, you should protect it. The built-in server supports API key authentication for both the REST endpoints and the web dashboard.

Set an API key using the environment variable:

```bash
export LONGTRACER_API_KEY="your-super-secret-key"
longtracer serve
```

When you visit `/dashboard`, you will be prompted to enter your API key to authenticate. Authentication relies on secure HTTP-only cookies and timing-safe digest comparisons.
