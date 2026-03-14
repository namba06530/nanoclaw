import mammoth from 'mammoth';
import { PDFParse } from 'pdf-parse';

const MAX_CHARS = 100_000;

export async function extractTextFromBuffer(
  buffer: Buffer,
  mimeType: string,
  filename: string,
): Promise<string> {
  try {
    let text = '';

    if (mimeType === 'application/pdf' || filename.endsWith('.pdf')) {
      const parser = new PDFParse({ data: new Uint8Array(buffer), verbosity: 0 });
      await (parser as any).load();
      const result = await parser.getText();
      text = result.text;
    } else if (
      mimeType ===
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
      filename.endsWith('.docx')
    ) {
      const result = await mammoth.extractRawText({ buffer });
      text = result.value;
    } else if (
      mimeType.startsWith('text/') ||
      ['.txt', '.md', '.csv', '.json'].some((ext) => filename.endsWith(ext))
    ) {
      text = buffer.toString('utf-8');
    } else {
      return `[Format non supporté pour l'extraction : ${filename}]`;
    }

    text = text.trim();

    // Detect scanned PDF: no text layer extracted
    if (
      (mimeType === 'application/pdf' || filename.endsWith('.pdf')) &&
      text.length < 100
    ) {
      return '[PDF scanné : ce document contient des images de texte, non du texte numérique. Aucun contenu extractible sans OCR.]';
    }

    if (text.length > MAX_CHARS) {
      return (
        text.slice(0, MAX_CHARS) +
        `\n\n[... contenu tronqué à ${MAX_CHARS} caractères]`
      );
    }
    return text || '[Fichier vide ou non lisible]';
  } catch (err: any) {
    return `[Erreur lors de l'extraction : ${err.message}]`;
  }
}
