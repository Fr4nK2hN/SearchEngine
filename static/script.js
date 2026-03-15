document.addEventListener("DOMContentLoaded", () => {
  const searchButton = document.getElementById("search-button");
  const searchBox = document.getElementById("search-box");
  const resultsContainer = document.getElementById("results-container");
  const modeSelector = document.getElementById("mode-selector");
  const modeRadios = document.querySelectorAll('input[name="mode"]');
  const visibleModeSelector = document.getElementById("visible-mode-selector");
  const loaderContainer = document.querySelector(".loader-container");
  const clearButton = document.getElementById("clear-button");
  const alertEl = document.getElementById("alert");
  const researchControls = document.getElementById("research-controls");
  const sessionInfo = document.getElementById("session-info");
  const toggleTrackingBtn = document.getElementById("toggle-tracking");
  const exportDataBtn = document.getElementById("export-data");
  const trackingStatus = document.getElementById("tracking-status");

  // Generate a unique session ID for this user session
  const sessionId =
    Date.now().toString(36) + Math.random().toString(36).substring(2);
  document.getElementById("session-id-display").textContent = sessionId;

  let events = [];
  let eventHistory = [];
  let lastQuery = null;
  let hasClickedResult = false;
  let isTrackingEnabled = true;
  let queryCount = 0;
  let clickCount = 0;
  let sessionStartTime = Date.now();
  let lastSearchTime = null;
  let pageVisibilityStart = Date.now();
  let currentSearchId = null;
  let isSendingEvents = false;
  let activeSendPromise = null;
  const pendingTrackClicks = new Set();

  // Check URL parameters for research mode
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get("research") === "true") {
    researchControls.style.display = "block";
    sessionInfo.style.display = "block";
  }

  // Sync visible mode selector with hidden one
  if (visibleModeSelector) {
    visibleModeSelector.addEventListener("change", () => {
      modeSelector.value = visibleModeSelector.value;
    });
  }

  // Sync radio controls with hidden selector (improved_frontend style)
  const syncModeFromRadios = () => {
    const selected = document.querySelector('input[name="mode"]:checked');
    if (selected) {
      modeSelector.value = selected.value;
    }
  };
  if (modeRadios && modeRadios.length) {
    syncModeFromRadios();
    modeRadios.forEach((r) => r.addEventListener("change", syncModeFromRadios));
  }

  // Toggle tracking functionality
  toggleTrackingBtn.addEventListener("click", () => {
    isTrackingEnabled = !isTrackingEnabled;
    trackingStatus.textContent = isTrackingEnabled ? "ON" : "OFF";
    trackingStatus.style.color = isTrackingEnabled ? "green" : "red";
  });

  // Export data functionality
  exportDataBtn.addEventListener("click", async () => {
    await sendEvents();
    if (pendingTrackClicks.size > 0) {
      await Promise.allSettled(Array.from(pendingTrackClicks));
    }

    let sessionData;
    try {
      const response = await fetch(
        `/export_data?session_id=${encodeURIComponent(sessionId)}`,
        { cache: "no-store" }
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      sessionData = await response.json();
      sessionData.client_session = {
        sessionId,
        sessionDuration: Date.now() - sessionStartTime,
        totalQueries: queryCount,
        totalClicks: clickCount,
        userAgent: navigator.userAgent,
      };
    } catch (error) {
      console.error("Failed to export server-side session data:", error);
      sessionData = {
        sessionId,
        sessionDuration: Date.now() - sessionStartTime,
        totalQueries: queryCount,
        totalClicks: clickCount,
        raw_events: eventHistory.slice(),
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString(),
        export_source: "client_fallback",
      };
    }

    const dataStr = JSON.stringify(sessionData, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `search_session_${sessionId}.json`;
    link.click();
    URL.revokeObjectURL(url);
  });

  // UI helpers
  const showAlert = (message) => {
    alertEl.textContent = message;
    alertEl.style.display = "block";
  };

  const clearAlert = () => {
    alertEl.textContent = "";
    alertEl.style.display = "none";
  };

  const escapeRegExp = (string) =>
    string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  const highlightText = (text, query) => {
    if (!query) return text;
    const terms = query
      .split(/\s+/)
      .filter((t) => t.length > 1)
      .map(escapeRegExp);
    if (terms.length === 0) return text;
    const regex = new RegExp(`(${terms.join("|")})`, "gi");
    return text.replace(regex, "<mark>$1</mark>");
  };

  // Enhanced event logging function
  const logEvent = (type, data) => {
    if (!isTrackingEnabled) return;

    const event = {
      type,
      sessionId,
      timestamp: new Date().toISOString(),
      timeFromStart: Date.now() - sessionStartTime,
      ...data,
    };

    events.push(event);
    eventHistory.push(event);

    // Update UI counters
    if (type === "query_submitted") {
      queryCount++;
      document.getElementById("query-count").textContent = queryCount;
    } else if (type === "result_clicked") {
      clickCount++;
      document.getElementById("click-count").textContent = clickCount;
    }
  };

  // Track page visibility changes
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      logEvent("page_hidden", {
        visibleDuration: Date.now() - pageVisibilityStart,
      });
    } else {
      pageVisibilityStart = Date.now();
      logEvent("page_visible", {});
    }
  });

  // Track scroll behavior
  let scrollTimeout;
  window.addEventListener("scroll", () => {
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
      logEvent("scroll_action", {
        scrollY: window.scrollY,
        scrollHeight: document.documentElement.scrollHeight,
        viewportHeight: window.innerHeight,
      });
    }, 500);
  });

  // Function to send events to the server
  const sendEvents = async () => {
    if (isSendingEvents) return activeSendPromise;
    if (events.length === 0) return;

    const batch = events.slice();
    isSendingEvents = true;
    activeSendPromise = (async () => {
      try {
        const response = await fetch("/log", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(batch),
          keepalive: true,
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        events = events.slice(batch.length);
      } catch (error) {
        console.error("Failed to send events:", error);
      } finally {
        isSendingEvents = false;
        activeSendPromise = null;
      }
    })();

    return activeSendPromise;
  };

  // Send events periodically
  setInterval(sendEvents, 15000); // Send every 15 seconds

  // Send events when the user leaves the page
  window.addEventListener("beforeunload", () => {
    // Track final session metrics
    logEvent("session_end", {
      totalDuration: Date.now() - sessionStartTime,
      finalQueryCount: queryCount,
      finalClickCount: clickCount,
    });
    if (events.length > 0 && navigator.sendBeacon) {
      const payload = new Blob([JSON.stringify(events)], {
        type: "application/json",
      });
      if (navigator.sendBeacon("/log", payload)) {
        events = [];
        return;
      }
    }
    sendEvents();
  });

  const performSearch = () => {
    const query = searchBox.value.trim();
    const mode =
      modeRadios && modeRadios.length
        ? document.querySelector('input[name="mode"]:checked')?.value ||
          modeSelector.value
        : modeSelector.value;

    if (query) {
      // Track query abandonment from the previous search
      if (lastQuery && !hasClickedResult) {
        logEvent("query_abandoned", {
          query: lastQuery,
          timeSpent: lastSearchTime ? Date.now() - lastSearchTime : 0,
        });
      }

      const searchStartTime = Date.now();
      lastSearchTime = searchStartTime;
      currentSearchId = null;

      logEvent("query_submitted", {
        query,
        mode,
        queryLength: query.length,
        isRepeatQuery: lastQuery === query,
      });

      lastQuery = query;
      hasClickedResult = false;

      // Show loader and clear previous results
      loaderContainer.style.display = "block";
      resultsContainer.innerHTML = "";
      clearAlert();
      resultsContainer.setAttribute("aria-busy", "true");
      searchButton.disabled = true;
      searchButton.textContent = "Searching...";

      const searchParams = new URLSearchParams({
        q: query,
        mode,
        session_id: sessionId,
      });

      fetch(`/search?${searchParams.toString()}`)
        .then((response) => response.json())
        .then((data) => {
          const searchEndTime = Date.now();
          loaderContainer.style.display = "none";
          resultsContainer.removeAttribute("aria-busy");
          searchButton.disabled = false;
          searchButton.textContent = "Search";
          currentSearchId = data.search_id || null;

          logEvent("search_completed", {
            query,
            mode,
            searchId: currentSearchId,
            resultCount: Array.isArray(data.results) ? data.results.length : 0,
            searchDuration: searchEndTime - searchStartTime,
            route_label: data.routing?.route_label || null,
            route_selected_mode: data.routing?.selected_mode || null,
            route_guardrail: data.routing?.route_guardrail || null,
            route_rerank_top_n: data.routing?.rerank_top_n || null,
            route_source: data.routing?.route_source || null,
            route_confidence:
              data.routing?.route_confidence !== undefined
                ? Number(data.routing.route_confidence)
                : null,
          });

          displayResults(data);
          void sendEvents();
        })
        .catch((error) => {
          console.error("Error fetching search results:", error);
          loaderContainer.style.display = "none";
          resultsContainer.removeAttribute("aria-busy");
          searchButton.disabled = false;
          searchButton.textContent = "Search";
          showAlert("Error loading results. Please try again.");

          logEvent("search_error", {
            query,
            error: error.message,
          });
        });
    }
  };

  searchButton.addEventListener("click", performSearch);
  searchBox.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      performSearch();
    }
  });

  // Track typing behavior
  let typingTimeout;
  searchBox.addEventListener("input", () => {
    clearTimeout(typingTimeout);
    typingTimeout = setTimeout(() => {
      if (searchBox.value.trim()) {
        logEvent("typing_pause", {
          currentQuery: searchBox.value.trim(),
          queryLength: searchBox.value.trim().length,
        });
      }
    }, 2000);
  });

  // Clear button logic
  clearButton.addEventListener("click", () => {
    searchBox.value = "";
    resultsContainer.innerHTML = "";
    clearAlert();
    loaderContainer.style.display = "none";
    resultsContainer.removeAttribute("aria-busy");
    logEvent("results_cleared", { query: lastQuery });
  });

  function displayResults(data) {
    const results = data.results;
    const routing = data.routing || null;

    const renderRoutingSummary = () => {
      if (!routing) return;
      const summary = document.createElement("div");
      summary.className = "routing-summary";

      const chips = [];
      if (routing.route_label) chips.push(`route: ${routing.route_label}`);
      if (routing.selected_mode) chips.push(`mode: ${routing.selected_mode}`);
      if (routing.rerank_top_n) chips.push(`top-k: ${routing.rerank_top_n}`);
      if (routing.route_confidence !== undefined) {
        const conf = Number(routing.route_confidence);
        if (!Number.isNaN(conf)) chips.push(`confidence: ${(conf * 100).toFixed(1)}%`);
      }
      if (routing.route_source) chips.push(`source: ${routing.route_source}`);

      chips.forEach((text) => {
        const chip = document.createElement("span");
        chip.className = "routing-chip";
        chip.textContent = text;
        summary.appendChild(chip);
      });

      resultsContainer.appendChild(summary);
    };

    if (!Array.isArray(results) || results.length === 0) {
      resultsContainer.innerHTML =
        '<div class="results-empty">No results found. Try refining your query.</div>';
      renderRoutingSummary();
      logEvent("serp_impression", {
        query: lastQuery,
        searchId: currentSearchId,
        results: [],
        resultCount: 0,
        route_label: routing?.route_label || null,
        route_selected_mode: routing?.selected_mode || null,
        route_guardrail: routing?.route_guardrail || null,
        route_rerank_top_n: routing?.rerank_top_n || null,
        route_source: routing?.route_source || null,
      });
      return;
    }

    // Log the entire SERP that was shown to the user
    const resultIds = results.map((r) => r._id);
    const resultScores = results.map((r) => r._score);

    logEvent("serp_impression", {
      query: lastQuery,
      searchId: currentSearchId,
      results: resultIds,
      resultCount: results.length,
      averageScore:
        resultScores.reduce((a, b) => a + b, 0) / resultScores.length,
      result_scores: resultScores,
      route_label: routing?.route_label || null,
      route_selected_mode: routing?.selected_mode || null,
      route_guardrail: routing?.route_guardrail || null,
      route_rerank_top_n: routing?.rerank_top_n || null,
      route_source: routing?.route_source || null,
    });

    // Function to optimize title display
    function optimizeTitle(originalTitle) {
      if (!originalTitle) return "Untitled";

      let title = originalTitle.trim();

      // Remove common prefixes that add no value
      const prefixesToRemove = [
        /^Definition of\s+/i,
        /^Medical Definition of\s+/i,
        /^Current local time in\s+/i,
        /^The name\s+\w+\s+is\s+/i,
        /^How rich is\s+/i,
        /^There are\s+\d+\s+calories in\s+/i,
        /^As of\s+\w+\s+\d{4},?\s+/i,
      ];

      prefixesToRemove.forEach((prefix) => {
        title = title.replace(prefix, "");
      });

      // Remove trailing periods and ellipsis
      title = title.replace(/\.+$/, "");

      // Remove phone numbers and addresses patterns
      title = title.replace(/Phone Number:\s*\([0-9\-\s]+\).*/i, "");
      title = title.replace(
        /\d{3,5}\s+[A-Za-z\s]+(?:St|Ave|Dr|Rd|Blvd|Lane|Way)\.?,?\s*[A-Z]{2}\s*\d{5}.*/i,
        ""
      );

      // Remove stock quotes and prices
      title = title.replace(
        /\$[\d,]+\.?\d*\s*[-+]?\$?[\d,]*\.?\d*\s*\([+-]?[\d.%]+\).*$/i,
        ""
      );

      // Remove excessive detail after certain patterns
      title = title.replace(/\s*-\s*\d+\s+people found this useful\.?$/i, "");
      title = title.replace(/\s*View More\.\.\.$/i, "");
      title = title.replace(/\s*See more\.$/i, "");
      title = title.replace(/\s*Learn more\.$/i, "");

      // Clean up multiple spaces
      title = title.replace(/\s+/g, " ").trim();

      // Capitalize first letter if needed
      if (title.length > 0) {
        title = title.charAt(0).toUpperCase() + title.slice(1);
      }

      // Limit length to reasonable size
      if (title.length > 80) {
        title = title.substring(0, 77) + "...";
      }

      // Fallback if title becomes empty
      if (!title || title.length < 3) {
        return originalTitle.length > 80
          ? originalTitle.substring(0, 77) + "..."
          : originalTitle;
      }

      return title;
    }

    results.forEach((result, index) => {
      const resultElement = document.createElement("div");
      resultElement.className = "result";
      resultElement.setAttribute("data-result-id", result._id);
      resultElement.setAttribute("data-rank", index + 1);

      const titleElement = document.createElement("h3");
      const titleLink = document.createElement("a");
      titleLink.href = "#";
      titleLink.textContent = optimizeTitle(result._source.title);

      // Enhanced click tracking
      titleLink.addEventListener("click", (e) => {
        e.preventDefault();

        logEvent("result_clicked", {
          query: lastQuery,
          searchId: currentSearchId,
          docId: result._id,
          rank: index + 1,
          score: result._score,
          timeFromSearch: lastSearchTime ? Date.now() - lastSearchTime : 0,
        });

        trackClick(currentSearchId, result._id, index + 1, lastQuery);
        void sendEvents();

        hasClickedResult = true;

        // Simulate navigation with more realistic behavior
        titleLink.style.color = "#14532d"; // Visited link color
        setTimeout(() => {
          alert(
            `Opening document: ${result._source.title.substring(0, 50)}...`
          );
        }, 100);
      });

      function trackClick(search_id, doc_id, rank, query) {
        const clickPromise = fetch("/track_click", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          keepalive: true,
          body: JSON.stringify({ session_id: sessionId, search_id, doc_id, rank, query }),
        }).catch((error) => {
          console.error("Failed to confirm click:", error);
        });
        pendingTrackClicks.add(clickPromise);
        clickPromise.finally(() => {
          pendingTrackClicks.delete(clickPromise);
        });
      }

      titleElement.appendChild(titleLink);

      const contentElement = document.createElement("p");
      const rawSnippet = result._source.content.substring(0, 300);
      const contentHtml = highlightText(rawSnippet, lastQuery) + "...";
      contentElement.innerHTML = contentHtml;

      // Add score display for research purposes (hidden by default)
      const scoreElement = document.createElement("small");
      scoreElement.textContent = `Score: ${result._score.toFixed(3)}`;
      scoreElement.style.color = "#666";
      scoreElement.style.display =
        urlParams.get("research") === "true" ? "block" : "none";

      resultElement.appendChild(titleElement);
      resultElement.appendChild(contentElement);
      resultElement.appendChild(scoreElement);

      // Track result visibility
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              logEvent("result_viewed", {
                query: lastQuery,
                docId: result._id,
                rank: index + 1,
                viewTime: Date.now(),
              });
              observer.unobserve(entry.target);
            }
          });
        },
        { threshold: 0.5 }
      );

      observer.observe(resultElement);
      if (index === 0) {
        renderRoutingSummary();
      }
      resultsContainer.appendChild(resultElement);
    });
  }

  // Log initial page load
  logEvent("session_start", {
    userAgent: navigator.userAgent,
    screenResolution: `${screen.width}x${screen.height}`,
    viewportSize: `${window.innerWidth}x${window.innerHeight}`,
    referrer: document.referrer,
  });
});
