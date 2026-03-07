import { motion } from "framer-motion";
import { Link } from "wouter";
import { ArrowRight, FlaskConical, Network, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const FLOATERS = Array.from({ length: 14 }).map((_, i) => ({
  id: i,
  size: 16 + (i % 5) * 10,
  left: `${6 + (i * 7) % 88}%`,
  top: `${8 + (i * 11) % 84}%`,
  delay: (i % 6) * 0.3,
}));

export default function Home() {
  return (
    <div className="relative min-h-[calc(100vh-3.5rem)] overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(59,130,246,0.16),transparent_40%),radial-gradient(circle_at_80%_30%,rgba(14,165,233,0.16),transparent_42%),radial-gradient(circle_at_50%_90%,rgba(16,185,129,0.12),transparent_36%)]" />

      {FLOATERS.map((f) => (
        <motion.div
          key={f.id}
          className="absolute rounded-full border border-primary/25 bg-primary/10"
          style={{ width: f.size, height: f.size, left: f.left, top: f.top }}
          animate={{ y: [0, -14, 0], opacity: [0.35, 0.8, 0.35], scale: [1, 1.15, 1] }}
          transition={{ duration: 4.2 + (f.id % 4), repeat: Infinity, ease: "easeInOut", delay: f.delay }}
        />
      ))}

      <div className="relative z-10 max-w-6xl mx-auto px-8 py-16 md:py-24">
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="uppercase tracking-[0.2em] text-xs text-primary/80 mb-5"
        >
          Electrochemical Intelligence Platform
        </motion.p>

        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.1 }}
          className="text-4xl md:text-7xl font-black leading-[1.02] max-w-5xl"
        >
          Accelerated Discovery of Electrochemical Energy Storage Systems
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mt-6 text-lg text-muted-foreground max-w-3xl"
        >
          Explore battery system design space with graph-based ML, live predictive inference, and large-scale
          materials catalogs. Move from target objectives to candidate systems in one workflow.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.3 }}
          className="mt-8 flex flex-wrap gap-3"
        >
          <Link href="/discovery">
            <Button size="lg" className="gap-2">
              Start Discovery <ArrowRight className="w-4 h-4" />
            </Button>
          </Link>
          <Link href="/prediction">
            <Button size="lg" variant="outline">
              Open Predictor
            </Button>
          </Link>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-4 mt-12">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Network className="w-4 h-4 text-primary" />
                Graph Intelligence
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              Masked GNN architecture for variable-component battery systems and objective-driven exploration.
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <FlaskConical className="w-4 h-4 text-primary" />
                Materials Scale
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              Large materials catalog integration with searchable formula + material ID selection.
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Zap className="w-4 h-4 text-primary" />
                Live Inference
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              Real-time discovery and prediction APIs with diagnostics, retries, auth controls, and runtime checks.
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

