"use client";
import Link from "next/link";
import { ArrowRight, LayoutDashboard, LogIn, Zap } from "lucide-react";
import { Section } from "@/components/ui/Section";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { StatCallout } from "@/components/ui/StatCallout";
import { SkeletonHero } from "@/components/visuals/SkeletonHero";
import { useAuth } from "@/contexts/AuthContext";

export function Hero() {
  const { doctor, loading } = useAuth();

  return (
    <Section className="relative overflow-hidden pt-40 md:pt-48 bg-grid">
      <div className="grid items-center gap-12 md:grid-cols-2 md:gap-16">
        <div>
          <Badge>AI-powered motion analysis</Badge>
          <h1 className="mt-6 text-hero">
            Gait and joints motion analysis
            <br />
            simplified<span className="text-accent">.</span>
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-muted">
            Extract clinical-grade biomechanics from any video.
            <br className="hidden sm:block" />
            No markers, no labs, no waiting room.
          </p>

          {/* CTA — auth-aware. Casual visitors are guided to sign up;
              already-logged-in doctors land on their dashboard. */}
          <div className="mt-10 flex flex-wrap items-center gap-3">
            {!loading && doctor ? (
              <Link href="/dashboard">
                <Button size="lg">
                  <LayoutDashboard className="h-4 w-4" />
                  Open dashboard
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
            ) : (
              <>
                <Link href="/auth/signup">
                  <Button size="lg">
                    Get started
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
                <Link href="/auth/signin">
                  <Button size="lg" variant="secondary">
                    <LogIn className="h-4 w-4" />
                    Sign in
                  </Button>
                </Link>
              </>
            )}
          </div>

          <div className="mt-14">
            <StatCallout value="30+" label="FPS realtime" icon={Zap} />
          </div>
        </div>

        <div className="relative aspect-square w-full max-w-[480px] justify-self-center md:max-w-none md:justify-self-end">
          <SkeletonHero />
        </div>
      </div>
    </Section>
  );
}
