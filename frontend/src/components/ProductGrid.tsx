'use client';

import ProductTile from './ProductTile';

interface ProductCard {
  id: string;
  name: string;
  brand: string;
  price: number;
  rating?: number;
  image_url?: string;
  description?: string;
  category?: string;
  subcategory?: string;
}

interface ProductGridProps {
  products: ProductCard[];
  title?: string;
}

export default function ProductGrid({ products, title }: ProductGridProps) {
  if (!products || products.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="text-gray-500 text-lg">No products found</div>
        <div className="text-gray-400 text-sm mt-1">Try adjusting your search criteria</div>
      </div>
    );
  }

  return (
    <div className="w-full">
      {title && (
        <div className="mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-2">{title}</h2>
          <div className="text-sm text-gray-600">
            Showing {products.length} product{products.length !== 1 ? 's' : ''}
          </div>
        </div>
      )}
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 md:gap-6">
        {products.map((product) => (
          <ProductTile key={product.id} product={product} />
        ))}
      </div>
    </div>
  );
}
