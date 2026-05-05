// Shared date/time formatter — always renders in Indian Standard Time
// (Asia/Kolkata, UTC+5:30) so timestamps line up with the doctor's clock
// regardless of the server's or browser's timezone.

const IST_OPTS: Intl.DateTimeFormatOptions = {
  timeZone: "Asia/Kolkata",
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "numeric",
  minute: "2-digit",
  hour12: true,
};

const IST_DATE_OPTS: Intl.DateTimeFormatOptions = {
  timeZone: "Asia/Kolkata",
  day: "numeric",
  month: "short",
  year: "numeric",
};

const IST_TIME_OPTS: Intl.DateTimeFormatOptions = {
  timeZone: "Asia/Kolkata",
  hour: "numeric",
  minute: "2-digit",
  hour12: true,
};

/** "5 May 2026, 11:58 AM" — date + time, IST. */
export function formatIST(iso: string | Date): string {
  const d = typeof iso === "string" ? new Date(iso) : iso;
  if (isNaN(d.getTime())) return "";
  return new Intl.DateTimeFormat("en-IN", IST_OPTS).format(d).replace(" at ", ", ");
}

/** "5 May 2026" — date only, IST. */
export function formatISTDate(iso: string | Date): string {
  const d = typeof iso === "string" ? new Date(iso) : iso;
  if (isNaN(d.getTime())) return "";
  return new Intl.DateTimeFormat("en-IN", IST_DATE_OPTS).format(d);
}

/** "11:58 AM" — time only, IST. */
export function formatISTTime(iso: string | Date): string {
  const d = typeof iso === "string" ? new Date(iso) : iso;
  if (isNaN(d.getTime())) return "";
  return new Intl.DateTimeFormat("en-IN", IST_TIME_OPTS).format(d);
}

/** "2026-05-05" — ISO-style YYYY-MM-DD in IST, for headers. */
export function formatISTIsoDate(iso: string | Date): string {
  const d = typeof iso === "string" ? new Date(iso) : iso;
  if (isNaN(d.getTime())) return "";
  // sv-SE locale outputs YYYY-MM-DD natively.
  return new Intl.DateTimeFormat("sv-SE", {
    timeZone: "Asia/Kolkata",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(d);
}
