// Global 404 handler — replaces Next.js's default plain "404" page
// with a navigable one so an unmatched URL (bad link, stale bookmark,
// URL typo, dev-server cache corruption) doesn't strand the user.

import Link from "next/link";
import { Compass, Home, Users } from "lucide-react";
import { Button } from "@/components/ui/Button";

export default function NotFound() {
  return (
    <main className="flex min-h-screen items-center justify-center px-6">
      <div className="max-w-md text-center">
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full border border-border bg-surface">
          <Compass className="h-6 w-6 text-muted" />
        </div>
        <h1 className="mt-6 text-3xl font-semibold tracking-tight">
          Page not found
        </h1>
        <p className="mt-3 text-sm text-muted">
          The link you followed doesn&apos;t match any page in this
          app. It may have been renamed, moved, or the URL is a
          typo.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <Link href="/dashboard">
            <Button variant="secondary">
              <Home className="h-4 w-4" />
              Dashboard
            </Button>
          </Link>
          <Link href="/dashboard/patients">
            <Button>
              <Users className="h-4 w-4" />
              Patients
            </Button>
          </Link>
        </div>
      </div>
    </main>
  );
}
