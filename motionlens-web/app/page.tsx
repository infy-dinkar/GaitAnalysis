import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Hero } from "@/components/landing/Hero";
import { Features } from "@/components/landing/Features";
import { ProductShowcase } from "@/components/landing/ProductShowcase";
import { UseCases } from "@/components/landing/UseCases";
import { FadeIn } from "@/components/ui/FadeIn";

export default function Home() {
  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Hero />
        <FadeIn>
          <Features />
        </FadeIn>
        <FadeIn>
          <ProductShowcase />
        </FadeIn>
        <FadeIn>
          <UseCases />
        </FadeIn>
      </main>
      <Footer />
    </>
  );
}
